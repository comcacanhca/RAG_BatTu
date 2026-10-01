"""
Embedding Service
- 使用 bge-large-en-v1.5 生成文本向量
- 支持 batch 编码
- BGE 系列模型 query 需要加 prefix
"""

import logging
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """文本 Embedding 服务"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda:0",
        batch_size: int = 64,
        max_seq_length: int = 512,
        normalize: bool = True,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prefix = query_prefix

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length

        # 获取向量维度
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def encode_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        编码文档（passages），不加 prefix
        返回: np.ndarray, shape=(n, dim)
        """
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码 query，BGE 模型需要加 prefix
        返回: np.ndarray, shape=(dim,)
        """
        # BGE 系列 query 需加 prefix
        prefixed_query = self.query_prefix + query
        embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=self.normalize,
        )
        return embedding[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """批量编码 queries"""
        prefixed = [self.query_prefix + q for q in queries]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
        )
        return embeddings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = Embedder(device="cuda:0")
    # 简单测试
    docs = ["This is a test document about machine learning.", "Deep learning is a subset of ML."]
    embs = embedder.encode_documents(docs, show_progress=False)
    print(f"Document embeddings shape: {embs.shape}")

    q_emb = embedder.encode_query("What is deep learning?")
    print(f"Query embedding shape: {q_emb.shape}")

    # 余弦相似度
    sims = embs @ q_emb
    print(f"Similarities: {sims}")
