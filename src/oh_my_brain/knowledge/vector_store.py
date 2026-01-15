"""向量存储模块.

提供文档嵌入和向量相似度搜索功能。
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """从字典创建."""
        embedding = None
        if data.get("embedding") is not None:
            embedding = np.array(data["embedding"], dtype=np.float32)
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=embedding,
        )


@dataclass
class SearchResult:
    """搜索结果."""

    document: Document
    score: float
    rank: int = 0


class EmbeddingProvider(ABC):
    """嵌入提供者基类."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入向量."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量生成嵌入向量."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入向量维度."""
        pass


class MiniMaxEmbedding(EmbeddingProvider):
    """MiniMax 嵌入服务.

    使用 MiniMax API 生成文本嵌入。
    """

    API_URL = "https://api.minimax.chat/v1/embeddings"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embo-01",
    ):
        """初始化.

        Args:
            api_key: API Key
            model: 模型名称
        """
        import os
        self._api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self._model = model
        self._dimension = 1536  # MiniMax embo-01 维度

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> np.ndarray:
        """生成嵌入向量."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量生成嵌入向量."""
        if not self._api_key:
            logger.warning("未配置 API Key，使用随机嵌入")
            return [self._random_embedding() for _ in texts]

        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "texts": texts,
            "type": "db",  # 存储用途
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            embeddings = []
            for item in data.get("vectors", []):
                embeddings.append(np.array(item, dtype=np.float32))

            return embeddings

        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            return [self._random_embedding() for _ in texts]

    def _random_embedding(self) -> np.ndarray:
        """生成随机嵌入（用于测试）."""
        return np.random.randn(self._dimension).astype(np.float32)


class LocalEmbedding(EmbeddingProvider):
    """本地嵌入模型.

    使用 sentence-transformers 生成嵌入。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """初始化.

        Args:
            model_name: 模型名称
        """
        self._model_name = model_name
        self._model = None
        self._dimension = 384  # all-MiniLM-L6-v2 默认维度

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self):
        """延迟加载模型."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                logger.warning("sentence-transformers 未安装，使用哈希嵌入")
                self._model = "hash"

    async def embed(self, text: str) -> np.ndarray:
        """生成嵌入向量."""
        self._load_model()

        if self._model == "hash":
            return self._hash_embedding(text)

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量生成嵌入向量."""
        self._load_model()

        if self._model == "hash":
            return [self._hash_embedding(t) for t in texts]

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.astype(np.float32) for e in embeddings]

    def _hash_embedding(self, text: str) -> np.ndarray:
        """基于哈希的简单嵌入（后备方案）."""
        # 使用 SHA256 生成确定性向量
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # 扩展到目标维度
        np.random.seed(int.from_bytes(hash_bytes[:4], "big"))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


class VectorStore(ABC):
    """向量存储基类."""

    @abstractmethod
    async def add(self, documents: list[Document]) -> None:
        """添加文档."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """搜索相似文档."""
        pass

    @abstractmethod
    async def delete(self, doc_ids: list[str]) -> None:
        """删除文档."""
        pass

    @abstractmethod
    async def get(self, doc_id: str) -> Document | None:
        """获取文档."""
        pass

    @abstractmethod
    def count(self) -> int:
        """文档数量."""
        pass


class InMemoryVectorStore(VectorStore):
    """内存向量存储.

    适用于中小规模知识库（<100K 文档）。
    """

    def __init__(self, persist_path: Path | None = None):
        """初始化.

        Args:
            persist_path: 持久化路径
        """
        self._documents: dict[str, Document] = {}
        self._embeddings: np.ndarray | None = None
        self._doc_ids: list[str] = []
        self._persist_path = persist_path
        self._dirty = False

        if persist_path and persist_path.exists():
            self._load()

    def count(self) -> int:
        return len(self._documents)

    async def add(self, documents: list[Document]) -> None:
        """添加文档."""
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"文档 {doc.id} 缺少嵌入向量")

            self._documents[doc.id] = doc

            if doc.id not in self._doc_ids:
                self._doc_ids.append(doc.id)

        # 重建嵌入矩阵
        self._rebuild_embeddings()
        self._dirty = True
        logger.info(f"添加 {len(documents)} 个文档")

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """搜索相似文档."""
        if self._embeddings is None or len(self._doc_ids) == 0:
            return []

        # 计算余弦相似度
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self._embeddings, query_norm)

        # 获取 top_k
        if filter_metadata:
            # 带过滤的搜索
            filtered_indices = []
            for i, doc_id in enumerate(self._doc_ids):
                doc = self._documents[doc_id]
                if self._match_metadata(doc.metadata, filter_metadata):
                    filtered_indices.append(i)

            if not filtered_indices:
                return []

            filtered_sims = similarities[filtered_indices]
            top_indices = np.argsort(filtered_sims)[::-1][:top_k]
            result_indices = [filtered_indices[i] for i in top_indices]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            result_indices = top_indices.tolist()

        results = []
        for rank, idx in enumerate(result_indices):
            doc_id = self._doc_ids[idx]
            results.append(SearchResult(
                document=self._documents[doc_id],
                score=float(similarities[idx]),
                rank=rank,
            ))

        return results

    async def delete(self, doc_ids: list[str]) -> None:
        """删除文档."""
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self._doc_ids.remove(doc_id)

        self._rebuild_embeddings()
        self._dirty = True
        logger.info(f"删除 {len(doc_ids)} 个文档")

    async def get(self, doc_id: str) -> Document | None:
        """获取文档."""
        return self._documents.get(doc_id)

    def _rebuild_embeddings(self) -> None:
        """重建嵌入矩阵."""
        if not self._doc_ids:
            self._embeddings = None
            return

        embeddings = []
        for doc_id in self._doc_ids:
            doc = self._documents[doc_id]
            if doc.embedding is not None:
                # 归一化
                norm = np.linalg.norm(doc.embedding)
                if norm > 0:
                    embeddings.append(doc.embedding / norm)
                else:
                    embeddings.append(doc.embedding)

        self._embeddings = np.vstack(embeddings) if embeddings else None

    def _match_metadata(
        self,
        doc_metadata: dict[str, Any],
        filter_metadata: dict[str, Any],
    ) -> bool:
        """检查元数据是否匹配."""
        for key, value in filter_metadata.items():
            if key not in doc_metadata:
                return False
            if isinstance(value, list):
                if doc_metadata[key] not in value:
                    return False
            elif doc_metadata[key] != value:
                return False
        return True

    def save(self, path: Path | None = None) -> None:
        """保存到文件."""
        save_path = path or self._persist_path
        if not save_path:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "doc_ids": self._doc_ids,
            "documents": [doc.to_dict() for doc in self._documents.values()],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        self._dirty = False
        logger.info(f"保存向量存储: {save_path}, {len(self._documents)} 文档")

    def _load(self) -> None:
        """从文件加载."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._doc_ids = data.get("doc_ids", [])
            for doc_data in data.get("documents", []):
                doc = Document.from_dict(doc_data)
                self._documents[doc.id] = doc

            self._rebuild_embeddings()
            logger.info(f"加载向量存储: {len(self._documents)} 文档")

        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")

    def __del__(self):
        """析构时自动保存."""
        if self._dirty and self._persist_path:
            self.save()


class FAISSVectorStore(VectorStore):
    """FAISS 向量存储.

    适用于大规模知识库（>100K 文档）。
    需要安装 faiss-cpu 或 faiss-gpu。
    """

    def __init__(
        self,
        dimension: int = 1536,
        persist_path: Path | None = None,
        use_gpu: bool = False,
    ):
        """初始化.

        Args:
            dimension: 向量维度
            persist_path: 持久化路径
            use_gpu: 是否使用 GPU
        """
        self._dimension = dimension
        self._persist_path = persist_path
        self._use_gpu = use_gpu

        self._documents: dict[str, Document] = {}
        self._doc_ids: list[str] = []
        self._index = None
        self._faiss = None

        self._init_faiss()

        if persist_path and persist_path.exists():
            self._load()

    def _init_faiss(self) -> None:
        """初始化 FAISS."""
        try:
            import faiss
            self._faiss = faiss

            # 创建索引
            self._index = faiss.IndexFlatIP(self._dimension)  # 内积（余弦相似度）

            if self._use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                    logger.info("FAISS GPU 索引已启用")
                except Exception:
                    logger.warning("无法使用 GPU，回退到 CPU")

        except ImportError:
            logger.warning("FAISS 未安装，使用内存存储作为后备")
            self._faiss = None

    def count(self) -> int:
        return len(self._documents)

    async def add(self, documents: list[Document]) -> None:
        """添加文档."""
        if self._faiss is None:
            raise RuntimeError("FAISS 未安装")

        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"文档 {doc.id} 缺少嵌入向量")

            self._documents[doc.id] = doc
            if doc.id not in self._doc_ids:
                self._doc_ids.append(doc.id)

            # 归一化
            norm = np.linalg.norm(doc.embedding)
            if norm > 0:
                embeddings.append(doc.embedding / norm)
            else:
                embeddings.append(doc.embedding)

        # 添加到 FAISS 索引
        embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        self._index.add(embeddings_matrix)

        logger.info(f"添加 {len(documents)} 个文档到 FAISS")

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """搜索相似文档."""
        if self._faiss is None or self._index.ntotal == 0:
            return []

        # 归一化查询向量
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_matrix = query_norm.reshape(1, -1).astype(np.float32)

        # 搜索更多以支持过滤
        search_k = top_k * 5 if filter_metadata else top_k
        scores, indices = self._index.search(query_matrix, min(search_k, len(self._doc_ids)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._doc_ids):
                continue

            doc_id = self._doc_ids[idx]
            doc = self._documents[doc_id]

            # 元数据过滤
            if filter_metadata and not self._match_metadata(doc.metadata, filter_metadata):
                continue

            results.append(SearchResult(
                document=doc,
                score=float(score),
                rank=len(results),
            ))

            if len(results) >= top_k:
                break

        return results

    async def delete(self, doc_ids: list[str]) -> None:
        """删除文档.

        注意：FAISS 不支持直接删除，需要重建索引。
        """
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self._doc_ids.remove(doc_id)

        # 重建索引
        await self._rebuild_index()
        logger.info(f"删除 {len(doc_ids)} 个文档")

    async def get(self, doc_id: str) -> Document | None:
        """获取文档."""
        return self._documents.get(doc_id)

    async def _rebuild_index(self) -> None:
        """重建 FAISS 索引."""
        if self._faiss is None:
            return

        self._index = self._faiss.IndexFlatIP(self._dimension)

        if self._doc_ids:
            embeddings = []
            for doc_id in self._doc_ids:
                doc = self._documents[doc_id]
                if doc.embedding is not None:
                    norm = np.linalg.norm(doc.embedding)
                    if norm > 0:
                        embeddings.append(doc.embedding / norm)
                    else:
                        embeddings.append(doc.embedding)

            if embeddings:
                embeddings_matrix = np.vstack(embeddings).astype(np.float32)
                self._index.add(embeddings_matrix)

    def _match_metadata(
        self,
        doc_metadata: dict[str, Any],
        filter_metadata: dict[str, Any],
    ) -> bool:
        """检查元数据是否匹配."""
        for key, value in filter_metadata.items():
            if key not in doc_metadata:
                return False
            if isinstance(value, list):
                if doc_metadata[key] not in value:
                    return False
            elif doc_metadata[key] != value:
                return False
        return True

    def save(self, path: Path | None = None) -> None:
        """保存."""
        save_path = path or self._persist_path
        if not save_path or self._faiss is None:
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        index_path = save_path.with_suffix(".faiss")
        self._faiss.write_index(self._index, str(index_path))

        # 保存文档数据
        data = {
            "doc_ids": self._doc_ids,
            "documents": [doc.to_dict() for doc in self._documents.values()],
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info(f"保存 FAISS 存储: {save_path}")

    def _load(self) -> None:
        """加载."""
        if not self._persist_path or self._faiss is None:
            return

        try:
            # 加载 FAISS 索引
            index_path = self._persist_path.with_suffix(".faiss")
            if index_path.exists():
                self._index = self._faiss.read_index(str(index_path))

            # 加载文档数据
            if self._persist_path.exists():
                with open(self._persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._doc_ids = data.get("doc_ids", [])
                for doc_data in data.get("documents", []):
                    doc = Document.from_dict(doc_data)
                    self._documents[doc.id] = doc

            logger.info(f"加载 FAISS 存储: {len(self._documents)} 文档")

        except Exception as e:
            logger.error(f"加载 FAISS 存储失败: {e}")


def create_vector_store(
    store_type: str = "memory",
    dimension: int = 1536,
    persist_path: Path | None = None,
    **kwargs,
) -> VectorStore:
    """创建向量存储.

    Args:
        store_type: 存储类型 (memory, faiss)
        dimension: 向量维度
        persist_path: 持久化路径
        **kwargs: 其他参数

    Returns:
        向量存储实例
    """
    if store_type == "faiss":
        return FAISSVectorStore(
            dimension=dimension,
            persist_path=persist_path,
            **kwargs,
        )
    else:
        return InMemoryVectorStore(persist_path=persist_path)


def create_embedding_provider(
    provider_type: str = "local",
    **kwargs,
) -> EmbeddingProvider:
    """创建嵌入提供者.

    Args:
        provider_type: 提供者类型 (local, minimax)
        **kwargs: 其他参数

    Returns:
        嵌入提供者实例
    """
    if provider_type == "minimax":
        return MiniMaxEmbedding(**kwargs)
    else:
        return LocalEmbedding(**kwargs)
