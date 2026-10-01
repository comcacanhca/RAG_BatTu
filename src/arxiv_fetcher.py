"""
arXiv Paper Fetcher
- 按 query / 类别 / 时间范围抓取论文
- 下载 PDF 到本地
"""

import os
import time
import arxiv
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class PaperMeta:
    """论文元数据"""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str
    updated: str
    pdf_url: str
    local_pdf_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ArxivFetcher:
    """arXiv 论文抓取器"""

    def __init__(
        self,
        download_dir: str = "data/pdfs",
        max_papers: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        categories: Optional[list[str]] = None,
    ):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_papers = max_papers
        self.categories = categories or []

        # 映射排序方式
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        order_map = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }
        self.sort_by = sort_map.get(sort_by, arxiv.SortCriterion.SubmittedDate)
        self.sort_order = order_map.get(sort_order, arxiv.SortOrder.Descending)

    def _build_query(self, query: str, broad: bool = False) -> str:
        """
        构建 arXiv 搜索 query，可带类别过滤。
        broad=False: 精确短语搜索（默认）
        broad=True:  宽泛搜索（拆词 OR，用于 fallback 补充召回）
        """
        if not broad:
            # 精确短语搜索: title 或 abstract 必须包含完整短语
            quoted = f'"{query}"'
            topic_filter = f"(ti:{quoted} OR abs:{quoted})"
        else:
            # 宽泛搜索: 拆词用 AND 搜索 all 字段
            words = query.strip().split()
            word_filters = " AND ".join(f"all:{w}" for w in words)
            topic_filter = f"({word_filters})"

        parts = [topic_filter]
        if self.categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in self.categories)
            parts.append(f"({cat_filter})")
        return " AND ".join(parts)

    @staticmethod
    def _normalize_arxiv_id(raw_id: str) -> str:
        """去掉版本号: '2301.00001v2' → '2301.00001'"""
        import re
        return re.sub(r"v\d+$", "", raw_id)

    def _search_once(
        self, query_str: str, max_papers: int, seen_ids: set
    ) -> list[PaperMeta]:
        """执行一次 arXiv 搜索，跳过 seen_ids 中已有的论文"""
        client = arxiv.Client(
            page_size=50,
            delay_seconds=3.0,
            num_retries=3,
        )
        search = arxiv.Search(
            query=query_str,
            max_results=max_papers,
            sort_by=self.sort_by,
            sort_order=self.sort_order,
        )

        papers = []
        for result in client.results(search):
            raw_id = result.entry_id.split("/")[-1]
            norm_id = self._normalize_arxiv_id(raw_id)

            if norm_id in seen_ids:
                continue
            seen_ids.add(norm_id)

            paper = PaperMeta(
                arxiv_id=norm_id,
                title=result.title.replace("\n", " ").strip(),
                authors=[a.name for a in result.authors],
                abstract=result.summary.replace("\n", " ").strip(),
                categories=result.categories,
                published=result.published.isoformat(),
                updated=result.updated.isoformat(),
                pdf_url=result.pdf_url,
            )
            papers.append(paper)

        return papers

    def search(
        self, query: str, max_papers: Optional[int] = None,
        fallback_threshold: int = 30,
    ) -> list[PaperMeta]:
        """
        搜索 arXiv 论文，返回元数据列表。
        1. 先精确短语搜索
        2. 如果结果 < fallback_threshold，自动用宽泛搜索补齐
        """
        max_papers = max_papers or self.max_papers
        seen_ids = set()

        # Round 1: 精确短语搜索
        precise_query = self._build_query(query, broad=False)
        logger.info(f"Precise search: '{precise_query}' (max={max_papers})")
        papers = self._search_once(precise_query, max_papers, seen_ids)
        logger.info(f"Precise search found {len(papers)} papers")

        # Round 2: 不够的话用宽泛搜索补齐
        if len(papers) < fallback_threshold:
            remaining = max_papers - len(papers)
            broad_query = self._build_query(query, broad=True)
            logger.info(f"Fallback broad search: '{broad_query}' (need {remaining} more)")
            extra = self._search_once(broad_query, remaining, seen_ids)
            papers.extend(extra)
            logger.info(f"Broad search added {len(extra)} papers, total={len(papers)}")

        logger.info(f"Found {len(papers)} unique papers total")
        return papers

    def _find_existing_pdf(self, safe_id: str) -> Optional[Path]:
        """查找已存在的 PDF（兼容旧版带 v1/v2 的文件名）"""
        # 精确匹配（新格式，不带版本号）
        exact = self.download_dir / f"{safe_id}.pdf"
        if exact.exists():
            return exact
        # 模糊匹配旧格式: 2602_21179v1.pdf, 2602_21179v2.pdf ...
        for f in self.download_dir.glob(f"{safe_id}v*.pdf"):
            return f  # 找到任意版本即可
        return None

    def download_pdfs(
        self, papers: list[PaperMeta], skip_existing: bool = True
    ) -> list[PaperMeta]:
        """
        批量下载 PDF，返回更新了 pdf_path 的元数据
        """
        downloaded = []
        for i, paper in enumerate(papers):
            # 文件名用 arxiv_id（把 . 和 / 替换掉）
            safe_id = paper.arxiv_id.replace("/", "_").replace(".", "_")
            pdf_path = self.download_dir / f"{safe_id}.pdf"

            # 兼容旧文件名（带 v1 等版本号）
            if skip_existing:
                existing = self._find_existing_pdf(safe_id)
                if existing:
                    logger.debug(f"[{i+1}/{len(papers)}] Skip existing: {existing.name}")
                    paper.local_pdf_path = str(existing)
                    downloaded.append(paper)
                    continue

            try:
                logger.info(
                    f"[{i+1}/{len(papers)}] Downloading: {paper.title[:60]}..."
                )
                # 使用 arxiv 库下载
                search = arxiv.Search(id_list=[paper.arxiv_id])
                client = arxiv.Client()
                result = next(client.results(search))
                result.download_pdf(dirpath=str(self.download_dir), filename=f"{safe_id}.pdf")

                paper.local_pdf_path = str(pdf_path)
                downloaded.append(paper)

                # Rate limit: arXiv 要求间隔 3 秒
                time.sleep(3)

            except Exception as e:
                logger.warning(f"Failed to download {paper.arxiv_id}: {e}")
                continue

        logger.info(f"Downloaded {len(downloaded)}/{len(papers)} PDFs")
        return downloaded

    def save_metadata(self, papers: list[PaperMeta], output_path: str = "data/papers_meta.json"):
        """保存元数据到 JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in papers]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata for {len(papers)} papers to {output_path}")

    @staticmethod
    def load_metadata(path: str) -> list[PaperMeta]:
        """从 JSON 加载元数据"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [PaperMeta(**d) for d in data]


# === 命令行测试 ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = ArxivFetcher(max_papers=5, categories=["cs.LG", "cs.CL"])
    papers = fetcher.search("diffusion models")
    for p in papers:
        print(f"  [{p.arxiv_id}] {p.title}")
    # papers = fetcher.download_pdfs(papers)
