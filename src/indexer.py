"""
Indexer: FAISS 向量索引 + BM25 稀疏索引
- FAISS: 纯本地，不需要 Docker 或外部服务
- BM25: rank-bm25，纯 Python
"""

import logging
import pickle
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    raise ImportError(
        "请安装 faiss:\n"
        "  pip install faiss-cpu   (CPU 版)\n"
        "  pip install faiss-gpu   (GPU 版，你有 RTX 6000 推荐这个)"
    )

from rank_bm25 import BM25Okapi


class FAISSIndexer:
    """
    FAISS 向量索引（替代 Qdrant，零外部依赖）
    
    使用 IndexFlatIP (Inner Product)，配合 normalized embeddings
    等价于 Cosine Similarity。数据量 < 10万时精确搜索完全够用。
    """

    def __init__(
        self,
        vector_size: int = 1024,
        index_type: str = "flat",   # "flat" 精确 | "ivf" 近似(数据量大时)
        nlist: int = 100,           # IVF 聚类数
        use_gpu: bool = False,
        save_dir: str = "data/faiss_index",
    ):
        self.vector_size = vector_size
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.save_dir = save_dir

        self.index = None
        self.chunks = []  # chunk 元数据，按 FAISS id 对齐

    def create_index(self, recreate: bool = False):
        """创建 FAISS 索引"""
        if self.index is not None and not recreate:
            logger.info("FAISS index already exists, skipping.")
            return

        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.vector_size)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.vector_size, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexFlatIP(self.vector_size)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving FAISS index to GPU")
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        self.chunks = []
        logger.info(f"Created FAISS index: type={self.index_type}, dim={self.vector_size}")

    def index_chunks(self, chunks: list[dict], embeddings: np.ndarray):
        """将 chunks + embeddings 写入 FAISS"""
        assert len(chunks) == len(embeddings), "chunks 和 embeddings 数量不匹配"
        logger.info(f"Indexing {len(chunks)} chunks into FAISS...")

        emb = np.ascontiguousarray(embeddings.astype(np.float32))

        # IVF 需要先 train
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(emb)

        self.index.add(emb)
        self.chunks = chunks
        logger.info(f"FAISS index complete. Total vectors: {self.index.ntotal}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filter_arxiv_id: Optional[str] = None,
        filter_method_family: Optional[list[str]] = None,
    ) -> list[dict]:
        """向量检索（FAISS 不原生支持 filter，用后过滤）"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty!")
            return []

        needs_filter = bool(filter_arxiv_id or filter_method_family)
        search_k = min(top_k * 4, self.index.ntotal) if needs_filter else top_k
        query = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))
        scores, indices = self.index.search(query, search_k)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if filter_arxiv_id and chunk.get("arxiv_id") != filter_arxiv_id:
                continue
            if filter_method_family and chunk.get("method_family", "") not in filter_method_family:
                continue
            hit = chunk.copy()
            hit["score"] = float(score)
            hit["point_id"] = int(idx)
            hits.append(hit)
            if len(hits) >= top_k:
                break

        return hits

    def save(self, dir_path: Optional[str] = None):
        """保存 FAISS 索引 + 元数据到磁盘"""
        dir_path = dir_path or self.save_dir
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(cpu_index, str(Path(dir_path) / "index.faiss"))

        with open(str(Path(dir_path) / "chunks_meta.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"FAISS index saved to {dir_path} ({self.index.ntotal} vectors)")

    def load(self, dir_path: Optional[str] = None):
        """从磁盘加载 FAISS 索引 + 元数据"""
        dir_path = dir_path or self.save_dir
        self.index = faiss.read_index(str(Path(dir_path) / "index.faiss"))

        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        with open(str(Path(dir_path) / "chunks_meta.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, {len(self.chunks)} chunks")

    def count(self) -> int:
        return self.index.ntotal if self.index else 0


class BM25Index:
    """BM25 稀疏索引"""

    # 学术论文高频停用词
    STOPWORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "need", "must",
        "it", "its", "this", "that", "these", "those", "he", "she", "they",
        "we", "i", "you", "me", "him", "her", "us", "them", "my", "your",
        "his", "our", "their", "which", "who", "whom", "what", "where",
        "when", "how", "why", "if", "then", "than", "so", "as", "not", "no",
        "nor", "also", "such", "more", "most", "very", "just", "about",
        "each", "other", "into", "over", "after", "before", "between",
        "through", "during", "above", "below", "all", "both", "same",
        "et", "al", "ie", "eg", "etc", "fig", "table", "eq", "ref",
    })

    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(chunks)} documents")

    def _tokenize(self, text: str) -> list[str]:
        import re
        text = text.lower()
        # 去标点，保留字母数字和连字符
        text = re.sub(r"[^a-z0-9\-\s]", " ", text)
        tokens = text.split()
        # 过滤停用词 + 过短token + 纯数字
        return [t for t in tokens
                if len(t) > 2 and t not in self.STOPWORDS and not t.isdigit()]

    def search(self, query: str, top_k: int = 20,
               filter_method_family: Optional[list[str]] = None) -> list[dict]:
        if self.bm25 is None:
            logger.error("BM25 index not built yet!")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        search_n = min(top_k * 4, len(self.chunks)) if filter_method_family else top_k
        top_indices = np.argsort(scores)[::-1][:search_n]

        hits = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            chunk = self.chunks[idx]
            if filter_method_family and chunk.get("method_family", "") not in filter_method_family:
                continue
            hit = chunk.copy()
            hit["bm25_score"] = float(scores[idx])
            hits.append(hit)
            if len(hits) >= top_k:
                break
        return hits

    def save(self, path: str = "data/bm25_index.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str = "data/bm25_index.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        logger.info(f"BM25 index loaded: {len(self.chunks)} documents")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 快速自测
    test_chunks = [
        {"chunk_id": "test_0", "text": "Diffusion models generate images by denoising.",
         "arxiv_id": "2301.00001", "title": "Test", "authors": [], "page_numbers": [1],
         "chunk_index": 0, "pdf_url": "", "published": ""},
        {"chunk_id": "test_1", "text": "Transformers use self-attention mechanism.",
         "arxiv_id": "2301.00002", "title": "Test2", "authors": [], "page_numbers": [1],
         "chunk_index": 0, "pdf_url": "", "published": ""},
    ]
    embeddings = np.random.randn(2, 1024).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    indexer = FAISSIndexer(vector_size=1024)
    indexer.create_index()
    indexer.index_chunks(test_chunks, embeddings)
    query = np.random.randn(1024).astype(np.float32)
    query = query / np.linalg.norm(query)
    results = indexer.search(query, top_k=2)
    for r in results:
        print(f"  FAISS score={r['score']:.4f}: {r['text'][:60]}")
