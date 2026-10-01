"""
Retriever: 混合检索 (Dense + Sparse) + Reranking
- Dense: FAISS 向量检索
- Sparse: BM25
- 融合: Reciprocal Rank Fusion (RRF)
- Rerank: bge-reranker-large
- 置信度评估: margin + mean_pos + num_pos
- 诊断日志: 每次 query 输出完整 JSON
"""

import logging
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Optional
from datetime import datetime
from sentence_transformers import CrossEncoder

from src.embedder import Embedder
from src.indexer import FAISSIndexer, BM25Index

logger = logging.getLogger(__name__)

# ─── Section-aware retrieval weights ─────────────────────────────────────────
# comparison: 需要方法细节和实验数据 → Method/Experiments 权重高
# taxonomy:   需要综述性描述 → Abstract/Intro/Related Work 权重高
# Section 名称来自 pdf_parser._detect_sections() 的归一化输出
_SECTION_WEIGHTS: dict[str, dict[str, float]] = {
    "comparison": {
        "Method":       1.30,
        "Experiments":  1.30,
        "Analysis":     1.20,   # ablation / discussion
        "Abstract":     1.00,
        "Introduction": 0.85,
        "Related Work": 0.80,   # background / preliminaries
        "Conclusion":   0.75,
    },
    "taxonomy": {
        "Abstract":     1.35,
        "Introduction": 1.25,
        "Related Work": 1.25,
        "Method":       1.10,
        "Analysis":     0.90,
        "Experiments":  0.85,
        "Conclusion":   1.00,
    },
}


class Reranker:
    """BGE Reranker"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cuda:0",
        batch_size: int = 32,
    ):
        logger.info(f"Loading reranker: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.batch_size = batch_size

    def rerank(self, query: str, documents: list[dict], top_k: int = 8) -> list[dict]:
        if not documents:
            return []
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_docs[:top_k]


class HybridRetriever:
    """混合检索器"""

    def __init__(
        self,
        embedder: Embedder,
        faiss_indexer: FAISSIndexer,
        bm25_index: BM25Index,
        reranker: Optional[Reranker] = None,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
        final_top_k: int = 20,
        rerank_top_k: int = 8,
        log_dir: str = "data/eval_logs",
    ):
        self.embedder = embedder
        self.faiss = faiss_indexer
        self.bm25 = bm25_index
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.rerank_top_k = rerank_top_k
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def retrieve(self, query: str, use_rerank: bool = True,
                 method_family_filter: Optional[list[str]] = None,
                 mode: str = "general") -> dict:
        """
        完整检索流程:
        1. Dense → 2. Sparse → 3. RRF → 4. 论文级去重(每篇≤2)
        → 5. Rerank → 6. 置信度评估(margin) → 7. 诊断日志

        method_family_filter: 若不为 None，只返回该 family 的 chunks（垂域过滤）
        """
        if method_family_filter:
            logger.info(f"method_family_filter active: {method_family_filter}")

        # ── Step 1: Dense retrieval ──
        query_emb = self.embedder.encode_query(query)
        dense_hits = self.faiss.search(query_emb, top_k=self.dense_top_k,
                                       filter_method_family=method_family_filter)
        logger.info(f"Dense retrieval: {len(dense_hits)} hits")

        # ── Step 2: Sparse retrieval ──
        sparse_hits = self.bm25.search(query, top_k=self.sparse_top_k,
                                       filter_method_family=method_family_filter)
        logger.info(f"Sparse retrieval: {len(sparse_hits)} hits")

        # ── Step 3: RRF fusion ──
        fused = self._rrf_fusion(dense_hits, sparse_hits)
        fused = fused[:self.final_top_k]
        logger.info(f"After RRF fusion: {len(fused)} candidates")

        # ── Step 3.5: Section-aware score boost ──────────────────────────────
        if mode != "general":
            fused = self._apply_section_boost(fused, mode)
            logger.info(f"Section boost applied (mode={mode})")

        # ── Step 4: 论文级去重（每篇最多 2 个 chunk，优先不同 section）──
        fused = self._dedup_by_paper(fused, max_per_paper=2)
        logger.info(f"After paper-level dedup: {len(fused)} candidates")

        # ── Step 5: Rerank ──
        if use_rerank and self.reranker:
            reranked = self.reranker.rerank(query, fused, top_k=self.rerank_top_k)
            logger.info(f"After reranking: {len(reranked)} results")
            docs = reranked
        else:
            docs = fused[:self.rerank_top_k]

        # ── Step 6: 置信度评估（margin + mean_pos）──
        confidence_info = self._assess_confidence(docs, use_rerank)
        logger.info(f"Retrieval confidence: {confidence_info['confidence']} "
                     f"(top1={confidence_info['top_score']:.2f}, "
                     f"margin={confidence_info['margin']:.2f}, "
                     f"mean_pos={confidence_info['mean_pos']:.2f}, "
                     f"relevant={confidence_info['num_relevant']}/{len(docs)})")

        # ── Step 7: 诊断日志 ──
        self._save_diagnostic_log(
            query=query,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_hits=fused,
            final_docs=docs,
            confidence_info=confidence_info,
        )

        return {
            "docs": docs,
            **confidence_info,
        }

    # ============================================================
    # #3: 置信度评估 — margin + mean_pos + num_pos
    # ============================================================
    def _assess_confidence(self, docs: list[dict], used_rerank: bool) -> dict:
        """
        基于 reranker logit 的多特征置信度评估:
          - top1:     最高分
          - margin:   top1 - top2（区分度）
          - mean_pos: score>0 的均值（整体相关度）
          - num_pos:  score>0 的数量

        分级规则（放宽 high 门槛，避免大部分 query 落入 medium 导致 prompt 弱化）:
          high:   top1 >= 0.5 AND num_pos >= 2
          medium: top1 > -1.0 OR num_pos >= 1
          low:    otherwise
        """
        if not docs:
            return {"confidence": "low", "top_score": -999.0, "margin": 0.0,
                    "mean_pos": 0.0, "num_relevant": 0}

        if used_rerank:
            scores = [d.get("rerank_score", 0.0) for d in docs]
        else:
            scores = [d.get("score", d.get("rrf_score", 0.0)) for d in docs]

        sorted_scores = sorted(scores, reverse=True)
        top1 = sorted_scores[0]
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else -999.0
        margin = top1 - top2

        pos_scores = [s for s in scores if s > 0]
        num_pos = len(pos_scores)
        mean_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0

        if top1 >= 0.5 and num_pos >= 2:
            confidence = "high"
        elif top1 > -1.0 or num_pos >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "confidence": confidence,
            "top_score": top1,
            "margin": margin,
            "mean_pos": mean_pos,
            "num_relevant": num_pos,
        }

    # ============================================================
    # #2: 论文级去重 — 每篇最多 2 个 chunk，优先不同 section
    # ============================================================
    @staticmethod
    def _dedup_by_paper(docs: list[dict], max_per_paper: int = 2) -> list[dict]:
        """
        每篇论文最多保留 max_per_paper 个 chunk。
        如果有 section 信息，优先保留不同 section 的 chunk。
        """
        paper_chunks = defaultdict(list)  # arxiv_id → [doc, ...]

        deduped = []
        for doc in docs:
            paper_id = doc.get("arxiv_id", doc.get("chunk_id", ""))
            existing = paper_chunks[paper_id]

            if len(existing) >= max_per_paper:
                continue

            paper_chunks[paper_id].append(doc)
            deduped.append(doc)

        return deduped

    # ============================================================
    # #0: 诊断日志 — 每次 query 输出完整 JSON
    # ============================================================
    def _save_diagnostic_log(
        self,
        query: str,
        dense_hits: list[dict],
        sparse_hits: list[dict],
        fused_hits: list[dict],
        final_docs: list[dict],
        confidence_info: dict,
    ):
        """保存每次 query 的完整诊断信息到 JSON"""

        def _hit_summary(hits: list[dict], score_key: str) -> list[dict]:
            """提取 hit 摘要"""
            summaries = []
            for rank, h in enumerate(hits):
                summaries.append({
                    "rank": rank + 1,
                    "arxiv_id": h.get("arxiv_id", ""),
                    "chunk_id": h.get("chunk_id", ""),
                    "score": round(h.get(score_key, 0.0), 4),
                    "section": h.get("section", ""),
                    "chunk_len": len(h.get("text", "")),
                })
            return summaries

        # 生成质量统计（如果有 answer 的话在 pipeline 层补充）
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieval": {
                "dense_top_k": _hit_summary(dense_hits[:self.dense_top_k], "score"),
                "sparse_top_k": _hit_summary(sparse_hits[:self.sparse_top_k], "bm25_score"),
                "rrf_fused": _hit_summary(fused_hits, "rrf_score"),
                "rerank_top8": _hit_summary(final_docs, "rerank_score"),
            },
            "confidence": {
                "level": confidence_info["confidence"],
                "top1": round(confidence_info["top_score"], 4),
                "margin": round(confidence_info["margin"], 4),
                "mean_pos": round(confidence_info["mean_pos"], 4),
                "num_pos": confidence_info["num_relevant"],
            },
        }

        # 追加写入 JSONL 日志文件
        log_path = self.log_dir / "query_diagnostics.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        logger.info(f"Diagnostic log saved to {log_path}")

    def _rrf_fusion(self, dense_hits: list[dict], sparse_hits: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank)"""
        scores = defaultdict(float)
        doc_map = {}

        for rank, hit in enumerate(dense_hits):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] += 1.0 / (self.rrf_k + rank + 1)
            doc_map[chunk_id] = hit

        for rank, hit in enumerate(sparse_hits):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] += 1.0 / (self.rrf_k + rank + 1)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = hit

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results = []
        for chunk_id in sorted_ids:
            doc = doc_map[chunk_id].copy()
            doc["rrf_score"] = scores[chunk_id]
            results.append(doc)
        return results

    def _apply_section_boost(self, docs: list[dict], mode: str) -> list[dict]:
        """
        Section-aware retrieval weighting:
        - comparison: Method/Experiments 权重 ×1.3，Abstract/Intro 降权
        - taxonomy:   Abstract/Introduction/Related Work 权重 ×1.35，Experiments 降权

        直接乘以 rrf_score，保持和 RRF 融合分数相同的量纲。
        """
        weights = _SECTION_WEIGHTS.get(mode, {})
        if not weights:
            return docs
        for doc in docs:
            section = doc.get("section", "").strip()
            w = weights.get(section, 1.0)
            if w != 1.0:
                doc["rrf_score"] = doc.get("rrf_score", 0.0) * w
        return sorted(docs, key=lambda d: d.get("rrf_score", 0.0), reverse=True)

    def dense_only(self, query: str, top_k: int = 10) -> list[dict]:
        """仅向量检索（用于消融实验）"""
        query_emb = self.embedder.encode_query(query)
        return self.faiss.search(query_emb, top_k=top_k)

    def sparse_only(self, query: str, top_k: int = 10) -> list[dict]:
        """仅 BM25 检索（用于消融实验）"""
        return self.bm25.search(query, top_k=top_k)
