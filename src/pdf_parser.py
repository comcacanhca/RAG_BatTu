"""
PDF Parser & Chunker
- PyMuPDF 提取文本
- RecursiveCharacterTextSplitter 分块
- Section 检测 + 标题/Section 前缀注入
- 多级噪声过滤（保留含关键指标的表格行）
"""

import os
import re
import json
import logging
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# 论文常见 section 名称（用于检测）
SECTION_PATTERNS = re.compile(
    r"^\s*(?:\d+\.?\s*)?("
    r"abstract|introduction|related\s*work|background|preliminary|preliminaries|"
    r"method|methods|methodology|approach|proposed\s*method|our\s*approach|"
    r"model|architecture|framework|"
    r"experiment|experiments|experimental\s*setup|experimental\s*results|"
    r"results|evaluation|analysis|ablation|discussion|"
    r"conclusion|conclusions|summary|limitation|limitations|"
    r"appendix|supplementary"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ─── 垂域 schema: LLM Alignment / Preference Optimization ───────────────
# 按优先级排列（先匹配者优先）
ALIGN_METHOD_FAMILIES: list[tuple[str, list[str]]] = [
    ("DPO",              ["direct preference optimization", "dpo"]),
    ("KTO",              ["kahneman-tversky", "kto"]),
    ("IPO",              ["identity preference optimization", "ipo"]),
    ("Constitutional AI",["constitutional ai", "cai"]),
    ("RLAIF",            ["reinforcement learning from ai feedback", "rlaif", "ai feedback"]),
    ("RLHF",             ["reinforcement learning from human feedback", "rlhf"]),
    ("PPO",              ["proximal policy optimization", "ppo"]),
    ("RM",               ["reward model", "reward modeling", "bradley-terry"]),
    ("SFT",              ["supervised fine-tuning", "instruction tuning"]),
    ("Offline RL",       ["best-of-n", "rejection sampling", "offline rl"]),
]

ALIGN_SIGNAL_TYPES: list[tuple[str, list[str]]] = [
    ("pairwise",  ["pairwise", "preference pair", "chosen", "rejected"]),
    ("binary",    ["binary feedback", "binary reward", "thumbs"]),
    ("ranking",   ["ranking", "ranked preference"]),
    ("critique",  ["critique", "self-critique", "feedback signal"]),
]

ALIGN_TRAIN_STAGES: list[tuple[str, list[str]]] = [
    ("SFT",        ["sft stage", "supervised fine-tuning stage", "warmup stage"]),
    ("RM",         ["reward model training", "preference modeling", "reward learning"]),
    ("policy-opt", ["policy optimization", "policy gradient", "ppo training", "dpo training"]),
]

ALIGN_TASKS: list[tuple[str, list[str]]] = [
    ("alignment",   ["value alignment", "align llm", "alignment objective"]),
    ("preference",  ["preference optimization", "preference learning", "human preference"]),
    ("safety",      ["harmless", "harmful", "refusal", "safety"]),
    ("evaluation",  ["evaluation", "benchmark", "win rate", "helpfulness"]),
]

# 含关键实验指标的行（不应被清洗掉）
METRIC_KEYWORDS = re.compile(
    r"\b(step|steps|nfe|fid|is\b|inception|speed|speedup|latency|"
    r"ms\b|sec\b|fps|throughput|accuracy|acc\b|bleu|rouge|psnr|ssim|"
    r"lpips|mse|mae|loss|epoch|iter|sample|sampling)\b",
    re.IGNORECASE,
)


@dataclass
class TextChunk:
    """文本块，带元数据"""
    chunk_id: str           # 唯一 ID: {arxiv_id}_{chunk_idx}
    arxiv_id: str
    title: str
    authors: list[str]
    text: str               # chunk 文本内容
    page_numbers: list[int] # 来自哪些页
    chunk_index: int        # 在该论文中的序号
    pdf_url: str
    published: str
    section: str = ""       # 所属 section（如 "Method", "Experiments"）
    # ── 垂域 schema (LLM Alignment / Preference Optimization) ────────────
    method_family: str = "" # DPO / PPO / RLHF / RLAIF / KTO / IPO / SFT / RM / etc.
    task: str = ""          # alignment / preference / safety / evaluation
    signal_type: str = ""   # pairwise / binary / ranking / critique
    train_stage: str = ""   # SFT / RM / policy-opt

    def to_dict(self) -> dict:
        return asdict(self)


class PDFParser:
    """PDF 解析 + 分块"""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_length: int = 80,
        output_dir: str = "data/parsed",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_chunk_length = min_chunk_length

        # LangChain 文本切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_text(self, pdf_path: str) -> tuple[str, list[tuple[int, str]]]:
        """
        从 PDF 提取文本
        返回: (全文文本, [(页码, 页文本), ...])
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            return "", []

        pages = []
        full_text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # 基础清理
            text = self._clean_text(text)

            if text.strip():
                pages.append((page_num + 1, text))
                full_text_parts.append(text)

        doc.close()
        full_text = "\n\n".join(full_text_parts)
        return full_text, pages

    def _clean_text(self, text: str) -> str:
        """
        清理 PDF 提取的文本。
        #5 改进：不再粗暴删除表格/公式行，而是保留含关键指标的行。
        """
        # 去掉多余空白行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 去掉页眉页脚中的页码模式
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        # 合并被换行打断的单词 (hyphenation)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # 去掉行尾多余空格
        text = re.sub(r"[ \t]+\n", "\n", text)
        # 去掉 LaTeX 残留 (如 \alpha, \begin{equation} 等)
        text = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", " ", text)

        # #5: 数学符号行 — 只删除不含关键指标的
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # 纯数学符号行检测
            is_math_line = bool(re.match(
                r"^[^\w]*[\u2200-\u22FF\u2190-\u21FF\u0391-\u03C9∑∏∫∂∇≤≥≠±×÷∈∉⊂⊃∀∃]",
                line
            ))
            # 纯数字/符号表格行检测
            is_table_line = bool(re.match(r"^[\d\s\.\,\-\+\|/±%]+$", line))

            if (is_math_line or is_table_line) and not METRIC_KEYWORDS.search(line):
                continue  # 删除不含关键指标的噪声行
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _detect_sections(self, full_text: str) -> list[tuple[int, str]]:
        """
        检测论文 section 标题及其在全文中的字符位置。
        返回: [(char_offset, section_name), ...]  按位置排序
        """
        sections = []
        for match in SECTION_PATTERNS.finditer(full_text):
            section_name = match.group(1).strip().title()
            # 合并近义词
            normalized = section_name.lower()
            if "method" in normalized or "approach" in normalized:
                section_name = "Method"
            elif "experiment" in normalized or "result" in normalized or "evaluation" in normalized:
                section_name = "Experiments"
            elif "related" in normalized or "background" in normalized or "preliminar" in normalized:
                section_name = "Related Work"
            elif "conclusion" in normalized or "summary" in normalized:
                section_name = "Conclusion"
            elif "abstract" in normalized:
                section_name = "Abstract"
            elif "introduction" in normalized:
                section_name = "Introduction"
            elif "ablation" in normalized or "analysis" in normalized or "discussion" in normalized:
                section_name = "Analysis"

            sections.append((match.start(), section_name))

        # 按位置排序
        sections.sort(key=lambda x: x[0])
        return sections

    def _get_section_for_position(self, char_pos: int, sections: list[tuple[int, str]]) -> str:
        """根据字符位置返回所属 section 名称"""
        current_section = ""
        for offset, name in sections:
            if offset > char_pos:
                break
            current_section = name
        return current_section

    @staticmethod
    def _extract_domain_schema(text: str) -> dict:
        """
        从文本中关键词匹配提取垂域 schema（无需 LLM，纯规则）。
        对每篇论文在入库时调用一次（用 title + abstract 即可）。
        """
        text_lower = text.lower()

        def _first_match(candidates: list[tuple[str, list[str]]]) -> str:
            for name, keywords in candidates:
                if any(kw in text_lower for kw in keywords):
                    return name
            return ""

        return {
            "method_family": _first_match(ALIGN_METHOD_FAMILIES),
            "task":          _first_match(ALIGN_TASKS),
            "signal_type":   _first_match(ALIGN_SIGNAL_TYPES),
            "train_stage":   _first_match(ALIGN_TRAIN_STAGES),
        }

    def chunk_paper(
        self,
        full_text: str,
        pages: list[tuple[int, str]],
        arxiv_id: str,
        title: str,
        authors: list[str],
        pdf_url: str,
        published: str,
        domain_schema: Optional[dict] = None,
    ) -> list[TextChunk]:
        """
        对单篇论文文本进行分块。
        #1 改进：chunk_size=1200, overlap=200, 注入 section 前缀
        """
        # 检测 section 结构
        sections = self._detect_sections(full_text)

        # 用 LangChain splitter 切分
        raw_chunks = self.splitter.split_text(full_text)

        # 为每个 chunk 找到在全文中的位置（用于确定 section）
        search_start = 0
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            # 过滤过短的 chunk
            if len(chunk_text.strip()) < self.min_chunk_length:
                continue

            # 过滤参考文献段
            lower = chunk_text.lower()
            if lower.lstrip().startswith("references") or lower.lstrip().startswith("bibliography"):
                continue
            # 跳过引用堆砌段 (如 "[12] A. Smith ... [13] B. Jones ...")
            bracket_refs = re.findall(r"\[\d+\]", chunk_text)
            if len(bracket_refs) >= 5 and len(chunk_text) < 800:
                continue
            # 过滤字母比例过低的 chunk（公式/表格残留）
            alpha_ratio = sum(c.isalpha() for c in chunk_text) / max(len(chunk_text), 1)
            if alpha_ratio < 0.35:
                continue

            # 确定 chunk 所属 section
            chunk_pos = full_text.find(chunk_text[:60], search_start)
            if chunk_pos >= 0:
                search_start = chunk_pos  # 避免重复匹配
            section = self._get_section_for_position(
                chunk_pos if chunk_pos >= 0 else 0, sections
            )

            # 找出 chunk 所在的页码
            page_nums = []
            for page_num, page_text in pages:
                snippet = chunk_text[:80].replace("\n", " ")
                if snippet[:40] in page_text.replace("\n", " "):
                    page_nums.append(page_num)
            if not page_nums:
                page_nums = [1]

            # 注入标题 + section 前缀
            if section:
                contextualized_text = f"[{title}] [Section: {section}] {chunk_text.strip()}"
            else:
                contextualized_text = f"[{title}] {chunk_text.strip()}"

            schema = domain_schema or {}
            chunk = TextChunk(
                chunk_id=f"{arxiv_id}_{idx:04d}",
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                text=contextualized_text,
                page_numbers=page_nums,
                chunk_index=idx,
                pdf_url=pdf_url,
                published=published,
                section=section,
                method_family=schema.get("method_family", ""),
                task=schema.get("task", ""),
                signal_type=schema.get("signal_type", ""),
                train_stage=schema.get("train_stage", ""),
            )
            chunks.append(chunk)

        return chunks

    def parse_paper(self, paper_meta: dict) -> list[TextChunk]:
        """
        解析单篇论文: PDF → 文本 → 分块
        paper_meta: PaperMeta.to_dict() 的结果
        """
        pdf_path = paper_meta.get("local_pdf_path", "")
        if not pdf_path or not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            return []

        arxiv_id = paper_meta["arxiv_id"]
        logger.info(f"Parsing: [{arxiv_id}] {paper_meta['title'][:60]}...")

        full_text, pages = self.extract_text(pdf_path)
        if not full_text:
            logger.warning(f"Empty text extracted from {pdf_path}")
            return []

        # 用 title + abstract（前 1500 字）做垂域 schema 抽取，比全文更精准
        abstract = paper_meta.get("abstract", "")
        schema_text = paper_meta["title"] + " " + abstract + " " + full_text[:1500]
        domain_schema = self._extract_domain_schema(schema_text)
        if domain_schema["method_family"]:
            logger.info(f"  → domain_schema: {domain_schema}")

        chunks = self.chunk_paper(
            full_text=full_text,
            pages=pages,
            arxiv_id=arxiv_id,
            title=paper_meta["title"],
            authors=paper_meta.get("authors", []),
            pdf_url=paper_meta.get("pdf_url", ""),
            published=paper_meta.get("published", ""),
            domain_schema=domain_schema,
        )

        logger.info(f"  → {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    def parse_all(self, papers: list[dict]) -> list[TextChunk]:
        """批量解析所有论文"""
        all_chunks = []
        for i, paper in enumerate(papers):
            logger.info(f"[{i+1}/{len(papers)}] Processing...")
            chunks = self.parse_paper(paper)
            all_chunks.extend(chunks)

        logger.info(f"Total: {len(all_chunks)} chunks from {len(papers)} papers")
        return all_chunks

    def save_chunks(self, chunks: list[TextChunk], output_path: Optional[str] = None):
        """保存 chunks 到 JSON"""
        if output_path is None:
            output_path = str(self.output_dir / "chunks.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    @staticmethod
    def load_chunks(path: str) -> list[TextChunk]:
        """从 JSON 加载 chunks"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [TextChunk(**d) for d in data]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = PDFParser(chunk_size=1200, chunk_overlap=200)
    print("PDF Parser ready.")
