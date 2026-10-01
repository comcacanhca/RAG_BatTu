"""
Pipeline: Day 0 端到端流程（纯本地版，不需要 Docker / vLLM）

用法:
  # 一键跑通: 抓取 + 建库 + 问答
  python pipeline.py --query "diffusion models" --max_papers 50 \
      --ask "What are the main approaches for accelerating diffusion model sampling?"

  # 只建库，不问问题（进入交互模式）
  python pipeline.py --query "diffusion models" --max_papers 50

  # 跳过抓取，用已有数据直接问答
  python pipeline.py --query "any" --skip_fetch \
      --ask "How does classifier-free guidance work?"
"""

import json
import re
import yaml
import logging
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from src.arxiv_fetcher import ArxivFetcher
from src.pdf_parser import PDFParser
from src.embedder import Embedder
from src.indexer import FAISSIndexer, BM25Index
from src.retriever import HybridRetriever, Reranker
from src.generator import (Generator, APIGenerator, format_context,
                           format_answer_for_display, analyze_generation_quality)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

# ─── AlignRAG: 问题意图检测 ──────────────────────────────────────────────────
_INTENT_PATTERNS: dict[str, re.Pattern] = {
    "comparison": re.compile(
        r"(?:^|\s)(vs\.?|versus)(?:\s|$)|"
        r"\b(compare|comparison|difference|differ|better|"
        r"advantage|trade.?off|when to use|which (?:is|are))\b",
        re.IGNORECASE),
    "taxonomy":   re.compile(
        r"\b(taxonomy|survey|landscape|categorize|classify|families|types of|"
        r"overview of|methods for|approaches to|techniques for|what (?:are|is) the)\b",
        re.IGNORECASE),
    "table":      re.compile(
        r"\b(table|extract|summarize|breakdown|details of|information from|"
        r"schema|fields|list the)\b",
        re.IGNORECASE),
}

_FAMILY_PATTERNS: dict[str, re.Pattern] = {
    "DPO":             re.compile(r"\bdpo\b", re.IGNORECASE),
    "PPO":             re.compile(r"\bppo\b", re.IGNORECASE),
    "RLHF":            re.compile(r"\brlhf\b", re.IGNORECASE),
    "RLAIF":           re.compile(r"\brlaif\b", re.IGNORECASE),
    "KTO":             re.compile(r"\bkto\b", re.IGNORECASE),
    "IPO":             re.compile(r"\bipo\b", re.IGNORECASE),
    "SFT":             re.compile(r"\bsft\b", re.IGNORECASE),
    "RM":              re.compile(r"\b(reward model|reward modeling)\b", re.IGNORECASE),
    "Constitutional AI": re.compile(r"\bconstitutional\b", re.IGNORECASE),
    "Offline RL":      re.compile(r"\b(best.of.n|rejection sampling|offline rl)\b", re.IGNORECASE),
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """端到端 RAG Pipeline（纯本地版）"""

    def __init__(self, config: dict):
        self.config = config
        self.embedder = None
        self.faiss_indexer = None
        self.bm25_index = None
        self.reranker = None
        self.retriever = None
        self.generator = None

    def init_models(self, skip_llm: bool = False):
        """初始化所有模型"""
        logger.info("=" * 60)
        logger.info("Initializing models...")
        logger.info("=" * 60)

        # 1. Embedding model (GPU 0)
        emb_cfg = self.config["embedding"]
        self.embedder = Embedder(
            model_name=emb_cfg["model_name"],
            device=emb_cfg["device"],
            batch_size=emb_cfg["batch_size"],
            max_seq_length=emb_cfg["max_seq_length"],
            normalize=emb_cfg["normalize"],
            query_prefix=emb_cfg.get("query_prefix", ""),
        )

        # 2. FAISS 索引
        faiss_cfg = self.config["faiss"]
        self.faiss_indexer = FAISSIndexer(
            vector_size=faiss_cfg["vector_size"],
            index_type=faiss_cfg.get("index_type", "flat"),
            use_gpu=faiss_cfg.get("use_gpu", False),
            save_dir=str(PROJECT_ROOT / faiss_cfg["save_dir"]),
        )

        # 3. BM25
        self.bm25_index = BM25Index()

        # 4. Reranker (GPU 0，和 embedding 共享)
        rr_cfg = self.config["reranker"]
        self.reranker = Reranker(
            model_name=rr_cfg["model_name"],
            device=rr_cfg["device"],
            batch_size=rr_cfg["batch_size"],
        )

        # 5. Retriever
        ret_cfg = self.config["retrieval"]
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            faiss_indexer=self.faiss_indexer,
            bm25_index=self.bm25_index,
            reranker=self.reranker,
            dense_top_k=ret_cfg["dense_top_k"],
            sparse_top_k=ret_cfg["sparse_top_k"],
            rrf_k=ret_cfg["rrf_k"],
            final_top_k=ret_cfg["final_top_k"],
            rerank_top_k=self.config["reranker"]["top_k"],
        )

        # 6. Generator (本地 HF 模型，GPU 1)
        if not skip_llm:
            llm_cfg = self.config["llm"]
            backend = llm_cfg.get("backend", "local")

            if backend == "api":
                # 用 API 后端 (vLLM / Ollama / DeepSeek 等)
                self.generator = APIGenerator(
                    api_base=llm_cfg["api_base"],
                    model=llm_cfg["model"],
                    api_key=llm_cfg.get("api_key", "not-needed"),
                    max_tokens=llm_cfg["max_new_tokens"],
                    temperature=llm_cfg["temperature"],
                    top_p=llm_cfg["top_p"],
                )
            else:
                # 本地 HuggingFace Transformers（默认）
                self.generator = Generator(
                    model_name=llm_cfg["model"],
                    device=llm_cfg["device"],
                    max_new_tokens=llm_cfg["max_new_tokens"],
                    temperature=llm_cfg["temperature"],
                    top_p=llm_cfg["top_p"],
                    torch_dtype=llm_cfg.get("torch_dtype", "float16"),
                )

        logger.info("All models initialized!")

    # ============================================================
    # Phase 1: Fetch & Parse
    # ============================================================
    def fetch_and_parse(self, query: str, max_papers: int = 50,
                        skip_existing: bool = True, clean_old: bool = False) -> list[dict]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 1: Fetching papers for query='{query}'")
        logger.info(f"{'='*60}")

        ax_cfg = self.config["arxiv"]
        pdf_dir = PROJECT_ROOT / ax_cfg["download_dir"]

        # 换 query 时可选清理旧 PDF，避免磁盘无限积累
        if clean_old and pdf_dir.exists():
            import shutil
            old_count = len(list(pdf_dir.glob("*.pdf")))
            shutil.rmtree(pdf_dir)
            pdf_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleaned {old_count} old PDFs from {pdf_dir}")

        fetcher = ArxivFetcher(
            download_dir=str(pdf_dir),
            max_papers=max_papers,
            sort_by=ax_cfg["sort_by"],
            sort_order=ax_cfg["sort_order"],
            categories=ax_cfg.get("categories"),
        )
        papers = fetcher.search(query, max_papers=max_papers)
        if not papers:
            logger.error("No papers found!")
            return []

        papers = fetcher.download_pdfs(papers, skip_existing=skip_existing)
        fetcher.save_metadata(papers, str(PROJECT_ROOT / "data/papers_meta.json"))

        pr_cfg = self.config["parser"]
        parser = PDFParser(
            chunk_size=pr_cfg["chunk_size"],
            chunk_overlap=pr_cfg["chunk_overlap"],
            min_chunk_length=pr_cfg["min_chunk_length"],
            output_dir=str(PROJECT_ROOT / pr_cfg["output_dir"]),
        )
        paper_dicts = [p.to_dict() for p in papers]
        chunks = parser.parse_all(paper_dicts)
        chunk_dicts = [c.to_dict() for c in chunks]

        # 把每篇论文的摘要也作为高质量 chunk 加入索引
        # 摘要是论文最好的概括，对综述性/概览性问题至关重要
        abstract_chunks = []
        for p in papers:
            if not p.abstract or len(p.abstract.strip()) < 50:
                continue
            abstract_chunks.append({
                "chunk_id": f"{p.arxiv_id}_abstract",
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors,
                "text": f"[{p.title}] Abstract: {p.abstract}",
                "page_numbers": [1],
                "chunk_index": -1,
                "pdf_url": p.pdf_url,
                "published": p.published,
            })
        chunk_dicts = abstract_chunks + chunk_dicts
        logger.info(f"Added {len(abstract_chunks)} abstract chunks")

        parser.save_chunks(chunks, str(PROJECT_ROOT / "data/parsed/chunks.json"))

        logger.info(f"Phase 1 complete: {len(papers)} papers → {len(chunk_dicts)} chunks (incl. abstracts)")
        return chunk_dicts

    # ============================================================
    # Phase 2: Index
    # ============================================================
    def build_index(self, chunk_dicts: list[dict], recreate: bool = True):
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 2: Building index for {len(chunk_dicts)} chunks")
        logger.info(f"{'='*60}")

        # Embedding
        texts = [c["text"] for c in chunk_dicts]
        embeddings = self.embedder.encode_documents(texts)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        np.save(str(PROJECT_ROOT / "data" / "embeddings.npy"), embeddings)

        # FAISS 建索引
        self.faiss_indexer.create_index(recreate=recreate)
        self.faiss_indexer.index_chunks(chunk_dicts, embeddings)
        self.faiss_indexer.save()

        # BM25 建索引
        self.bm25_index.build(chunk_dicts)
        self.bm25_index.save(str(PROJECT_ROOT / "data/bm25_index.pkl"))

        logger.info("Phase 2 complete: Index built!")

    # ============================================================
    # AlignRAG: Query Planning（Agentic Step 1）
    # ============================================================
    def _plan_sub_queries(self, question: str, mode: str) -> list[str]:
        """
        用 LLM 把问题分解成 2-3 个检索子查询。
        只在 comparison / taxonomy 模式下触发；无 generator 或 planner 被禁用时退化为原始查询。
        """
        if (mode not in ("comparison", "taxonomy")
                or not self.generator
                or not self.config.get("agents", {}).get("query_planner", True)):
            return [question]
        return self.generator.plan_queries(question, mode)

    # ============================================================
    # AlignRAG: 问题意图检测
    # ============================================================
    @staticmethod
    def _classify_query_intent(question: str) -> tuple[str, list[str] | None]:
        """
        检测问题意图和涉及的 method family。
        Returns:
            mode: "taxonomy" | "comparison" | "table" | "general"
            family_filter: list[str] to filter by, or None (不过滤)
        """
        mode = "general"
        for m, pattern in _INTENT_PATTERNS.items():
            if pattern.search(question):
                mode = m
                break

        families = [fam for fam, pat in _FAMILY_PATTERNS.items() if pat.search(question)]
        return mode, (families if families else None)

    # ============================================================
    # Phase 3: Query
    # ============================================================
    def query(self, question: str, use_rerank: bool = True) -> dict:
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 3: Answering question")
        logger.info(f"{'='*60}")

        mode, family_filter = self._classify_query_intent(question)
        # taxonomy 要求覆盖所有 family，过滤会把库搜空；只在 comparison/general 时过滤
        effective_filter = None if mode == "taxonomy" else family_filter
        if mode != "general" or effective_filter:
            logger.info(f"Query intent: mode={mode}, family_filter={effective_filter}")

        # ── Cross-Query Memory: warm-start boost ─────────────────────────────
        memory = self._load_query_memory()
        memory_boost_papers = self._find_memory_boost_papers(question, mode, memory)

        # ── Agentic Step 1: Query Planning ──────────────────────────────────
        sub_queries = self._plan_sub_queries(question, mode)

        # ── Multi-query retrieval + merge ────────────────────────────────────
        all_docs: list[dict] = []
        seen_chunk_ids: set[str] = set()
        best_result: dict = {}

        for sq in sub_queries:
            r = self.retriever.retrieve(
                sq, use_rerank=use_rerank,
                method_family_filter=effective_filter, mode=mode)
            for doc in r["docs"]:
                cid = doc.get("chunk_id", "")
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_docs.append(doc)
            if not best_result or r["top_score"] > best_result.get("top_score", -999.0):
                best_result = r

        if len(sub_queries) > 1:
            logger.info(f"Multi-query merge: {len(all_docs)} unique docs from {len(sub_queries)} sub-queries")
            # 重新按 rerank_score（或 rrf_score）排序
            all_docs.sort(
                key=lambda d: d.get("rerank_score", d.get("rrf_score", 0.0)),
                reverse=True,
            )

        retrieved = all_docs
        confidence = best_result.get("confidence", "low")
        top_score = best_result.get("top_score", 0.0)
        num_relevant = best_result.get("num_relevant", 0)
        retrieval_result = best_result

        if not retrieved:
            return {"question": question, "answer": "No relevant documents found.", "citations": [],
                    "confidence": "low"}

        agents_cfg = self.config.get("agents", {})

        # ── Agentic Step 2: Gap Detection ────────────────────────────────────
        gap_did_supplement = False
        if (self.generator and self.retriever and mode in ("comparison", "taxonomy")
                and agents_cfg.get("gap_detector", True)):
            gap = self.generator.check_gaps(question, mode, retrieved)
            if not gap["covered"] and gap["supplement_queries"]:
                logger.info(f"Gap detected → supplemental queries: {gap['supplement_queries']}")
                for sq in gap["supplement_queries"]:
                    r = self.retriever.retrieve(
                        sq, use_rerank=use_rerank,
                        method_family_filter=effective_filter, mode=mode)
                    for doc in r["docs"]:
                        cid = doc.get("chunk_id", "")
                        if cid not in seen_chunk_ids:
                            seen_chunk_ids.add(cid)
                            all_docs.append(doc)
                all_docs.sort(
                    key=lambda d: d.get("rerank_score", d.get("rrf_score", 0.0)),
                    reverse=True)
                retrieved = all_docs
                gap_did_supplement = True
                logger.info(f"After gap-fill: {len(retrieved)} unique docs")

        # ── Cross-Query Memory: boost docs from similar past queries ─────────
        if memory_boost_papers:
            for doc in retrieved:
                if doc.get("arxiv_id") in memory_boost_papers:
                    doc["_mem_bias"] = 0.5   # additive bias on rerank/rrf score
            retrieved.sort(
                key=lambda d: (d.get("rerank_score", d.get("rrf_score", 0.0))
                               + d.get("_mem_bias", 0.0)),
                reverse=True)
            logger.info(f"Memory boost: promoted {len(memory_boost_papers)} known-good papers")

        # 多子查询时适当扩大 context 窗口（但不超过 8）
        # gap-fill 额外补 2 个 slot，确保补充检索的 doc 能进入 context
        num_citations = self.config["llm"]["num_citations"]
        max_cit = min(num_citations * len(sub_queries), 8) if len(sub_queries) > 1 else num_citations
        if gap_did_supplement:
            max_cit = min(max_cit + 2, 8)

        if self.generator:
            result = self.generator.generate(
                question=question,
                retrieved_docs=retrieved,
                max_citations=max_cit,
                confidence=confidence,
                mode=mode,
            )

            # ── Agentic Step 3: Self-Critique ────────────────────────────────
            if (mode in ("comparison", "taxonomy")
                    and agents_cfg.get("self_critique", True)):
                context_str, _ = format_context(retrieved, max_docs=max_cit)
                revised = self.generator.critique(
                    question=question,
                    mode=mode,
                    context_str=context_str,
                    answer=result["answer"],
                    num_docs=result["num_citations"],
                )
                if revised:
                    result["answer"] = revised
                    result["answer_revised"] = True
                    logger.info("Answer revised by self-critique agent")

            # ── Cross-Query Memory: save cited papers for future warm-start ──
            cited_ids = [c["arxiv_id"] for c in result.get("citations", [])]
            self._save_memory_entry(question, mode, cited_ids)
        else:
            result = {
                "question": question,
                "answer": "[LLM not loaded] Top retrieved passages:\n" +
                          "\n---\n".join(d["text"][:200] for d in retrieved[:5]),
                "citations": [
                    {"index": i+1, "title": d["title"], "arxiv_id": d["arxiv_id"],
                     "abs_url": f"https://arxiv.org/abs/{d['arxiv_id']}", "pdf_url": d.get("pdf_url", "")}
                    for i, d in enumerate(retrieved[:5])
                ],
            }

        # 附加 agentic pipeline 追踪信息（供 UI / eval 使用）
        result["mode"] = mode
        result["sub_queries"] = sub_queries
        result["gap_supplemented"] = gap_did_supplement

        # 附加检索诊断信息
        result["confidence"] = confidence
        result["retrieval_top_score"] = top_score
        result["retrieval_margin"] = retrieval_result.get("margin", 0.0)
        result["retrieval_mean_pos"] = retrieval_result.get("mean_pos", 0.0)
        result["retrieval_num_relevant"] = num_relevant

        # 生成质量统计
        gen_stats = analyze_generation_quality(result)
        result["generation_stats"] = gen_stats
        logger.info(f"Generation stats: tokens={gen_stats['answer_tokens']}, "
                     f"citations_used={gen_stats['num_citations_used']}, "
                     f"coverage={gen_stats['citation_coverage']:.0%}, "
                     f"uncited={gen_stats['uncited_sentences']}")

        # 追加到诊断日志（和 retriever 的日志放一起）
        self._append_generation_log(question, result, gen_stats)

        return result

    def _append_generation_log(self, question: str, result: dict, gen_stats: dict):
        """把生成质量统计追加到诊断日志"""
        log_dir = PROJECT_ROOT / "data" / "eval_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "generation_diagnostics.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "confidence": result.get("confidence"),
            "retrieval_top_score": round(result.get("retrieval_top_score", 0), 4),
            "retrieval_margin": round(result.get("retrieval_margin", 0), 4),
            "retrieval_mean_pos": round(result.get("retrieval_mean_pos", 0), 4),
            "retrieval_num_relevant": result.get("retrieval_num_relevant", 0),
            "generation": gen_stats,
            "citations": [
                {"index": c["index"], "title": c["title"], "arxiv_id": c["arxiv_id"]}
                for c in result.get("citations", [])
            ],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ============================================================
    # Cross-Query Memory Cache
    # ============================================================
    _MEMORY_STOPWORDS = frozenset({
        "the", "a", "an", "of", "in", "for", "and", "or", "is", "are",
        "was", "were", "what", "how", "which", "when", "where", "does",
        "do", "vs", "versus", "between", "compare", "comparison",
        "difference", "differences", "with",
    })

    def _load_query_memory(self) -> list[dict]:
        """加载历史查询记忆"""
        path = PROJECT_ROOT / "data" / "query_memory.json"
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _find_memory_boost_papers(self, question: str, mode: str,
                                   memory: list[dict]) -> set[str]:
        """
        找到与当前问题相似的历史查询（同 mode + key_terms Jaccard ≥ 0.4），
        返回那些查询中被引用过的 arxiv_id 集合，用于 warm-start boost。
        """
        if not memory or mode == "general":
            return set()
        cur_terms = {w.lower() for w in re.findall(r'\w+', question)
                     if w.lower() not in self._MEMORY_STOPWORDS and len(w) > 2}
        boosted: set[str] = set()
        for entry in memory:
            if entry.get("mode") != mode:
                continue
            past_terms = set(entry.get("key_terms", []))
            if not past_terms or not cur_terms:
                continue
            jaccard = len(cur_terms & past_terms) / len(cur_terms | past_terms)
            if jaccard >= 0.4:
                ids = entry.get("cited_arxiv_ids", [])
                boosted.update(ids)
                logger.info(f"Memory match (J={jaccard:.2f}): '{entry['question'][:60]}' "
                            f"→ boost {len(ids)} papers")
        return boosted

    def _save_memory_entry(self, question: str, mode: str, cited_arxiv_ids: list[str]):
        """把当前 query 的引用结果存入记忆（general 模式或无引用时跳过）"""
        if not cited_arxiv_ids or mode == "general":
            return
        path = PROJECT_ROOT / "data" / "query_memory.json"
        entries = self._load_query_memory()
        terms = sorted({w.lower() for w in re.findall(r'\w+', question)
                        if w.lower() not in self._MEMORY_STOPWORDS and len(w) > 2})
        entries.append({
            "question": question,
            "mode": mode,
            "key_terms": terms,
            "cited_arxiv_ids": cited_arxiv_ids,
            "timestamp": datetime.now().isoformat(),
        })
        entries = entries[-50:]   # 最多保留 50 条，滚动淘汰
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")

    # ============================================================
    # 加载已有索引
    # ============================================================
    def load_existing_index(self):
        """加载 FAISS + BM25 索引"""
        faiss_dir = PROJECT_ROOT / self.config["faiss"]["save_dir"]
        bm25_path = PROJECT_ROOT / "data/bm25_index.pkl"

        if (faiss_dir / "index.faiss").exists():
            self.faiss_indexer.load(str(faiss_dir))
        else:
            logger.warning(f"FAISS index not found at {faiss_dir}")

        if bm25_path.exists():
            self.bm25_index.load(str(bm25_path))
        else:
            logger.warning(f"BM25 index not found at {bm25_path}")


def main():
    parser = argparse.ArgumentParser(description="arXiv RAG QA Pipeline (Local)")
    parser.add_argument("--query", type=str, required=True, help="arXiv search query")
    parser.add_argument("--max_papers", type=int, default=50, help="Max papers to fetch")
    parser.add_argument("--ask", type=str, default=None, help="Question to ask")
    parser.add_argument("--skip_fetch", action="store_true", help="Skip fetching, use existing data")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM (retrieval only)")
    parser.add_argument("--clean", action="store_true", help="Clean old PDFs before fetching (use when switching query)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    args = parser.parse_args()
    config = load_config(args.config)
    pipeline = RAGPipeline(config)
    pipeline.init_models(skip_llm=args.skip_llm)

    if not args.skip_fetch:
        chunk_dicts = pipeline.fetch_and_parse(
            query=args.query, max_papers=args.max_papers, clean_old=args.clean)
        if chunk_dicts:
            pipeline.build_index(chunk_dicts)
    else:
        pipeline.load_existing_index()

    if args.ask:
        result = pipeline.query(args.ask)
        print(format_answer_for_display(result))
    else:
        print("\n" + "=" * 60)
        print("📚 arXiv RAG QA System Ready!")
        print("Type your question (or 'quit' to exit):")
        print("=" * 60)
        while True:
            try:
                question = input("\n❓ Question: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    break
                if not question:
                    continue
                result = pipeline.query(question)
                print(format_answer_for_display(result))
            except KeyboardInterrupt:
                break
        print("\nBye! 👋")


if __name__ == "__main__":
    main()
