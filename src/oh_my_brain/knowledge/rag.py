"""RAG 检索增强生成引擎.

提供基于知识库的上下文检索，增强 LLM 生成能力。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oh_my_brain.knowledge.vector_store import (
    Document,
    EmbeddingProvider,
    SearchResult,
    VectorStore,
    create_embedding_provider,
    create_vector_store,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """RAG 上下文."""

    query: str
    retrieved_docs: list[SearchResult]
    formatted_context: str
    metadata: dict[str, Any]

    @property
    def has_context(self) -> bool:
        """是否有相关上下文."""
        return len(self.retrieved_docs) > 0

    @property
    def top_score(self) -> float:
        """最高相关度分数."""
        if self.retrieved_docs:
            return self.retrieved_docs[0].score
        return 0.0


@dataclass
class ChunkConfig:
    """文本分块配置."""

    chunk_size: int = 500       # 块大小（字符数）
    chunk_overlap: int = 100    # 重叠大小
    min_chunk_size: int = 50    # 最小块大小
    separators: list[str] | None = None  # 分隔符列表


class TextChunker:
    """文本分块器."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "]

    def __init__(self, config: ChunkConfig | None = None):
        """初始化.

        Args:
            config: 分块配置
        """
        self._config = config or ChunkConfig()
        self._separators = self._config.separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """将文本分块.

        Args:
            text: 输入文本
            metadata: 元数据

        Returns:
            文档块列表
        """
        if not text.strip():
            return []

        chunks = self._split_recursive(text, self._separators)
        documents = []

        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < self._config.min_chunk_size:
                continue

            doc_metadata = dict(metadata) if metadata else {}
            doc_metadata["chunk_index"] = i
            doc_metadata["chunk_total"] = len(chunks)

            documents.append(Document(
                id=f"chunk-{i}",
                content=chunk_text.strip(),
                metadata=doc_metadata,
            ))

        return documents

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """递归分割文本."""
        if not separators:
            return self._split_by_size(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(separator)
        chunks = []
        current_chunk = ""

        for part in parts:
            if not part.strip():
                continue

            potential_chunk = current_chunk + separator + part if current_chunk else part

            if len(potential_chunk) <= self._config.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if len(part) > self._config.chunk_size and remaining_separators:
                    # 递归分割大块
                    sub_chunks = self._split_recursive(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        # 处理重叠
        if self._config.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_by_size(self, text: str) -> list[str]:
        """按大小分割."""
        chunks = []
        for i in range(0, len(text), self._config.chunk_size - self._config.chunk_overlap):
            chunk = text[i:i + self._config.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """添加重叠."""
        if len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # 从前一个块末尾取重叠部分
                prev_chunk = chunks[i - 1]
                overlap = prev_chunk[-self._config.chunk_overlap:]
                chunk = overlap + chunk

            overlapped.append(chunk)

        return overlapped


class RAGEngine:
    """RAG 检索增强生成引擎.

    功能：
    1. 知识库索引和检索
    2. 上下文构建
    3. 与 LLM 集成
    4. 知识更新
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        persist_dir: Path | None = None,
    ):
        """初始化.

        Args:
            vector_store: 向量存储
            embedding_provider: 嵌入提供者
            persist_dir: 持久化目录
        """
        self._persist_dir = persist_dir

        # 初始化组件
        if embedding_provider:
            self._embedding = embedding_provider
        else:
            self._embedding = create_embedding_provider("local")

        if vector_store:
            self._store = vector_store
        else:
            store_path = persist_dir / "vectors.json" if persist_dir else None
            self._store = create_vector_store("memory", persist_path=store_path)

        self._chunker = TextChunker()

    @property
    def document_count(self) -> int:
        """文档数量."""
        return self._store.count()

    async def index_text(
        self,
        text: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """索引文本.

        Args:
            text: 文本内容
            source_id: 来源 ID
            metadata: 元数据

        Returns:
            索引的块数量
        """
        # 分块
        base_metadata = dict(metadata) if metadata else {}
        base_metadata["source_id"] = source_id

        chunks = self._chunker.chunk(text, base_metadata)

        if not chunks:
            return 0

        # 更新 ID
        for i, chunk in enumerate(chunks):
            chunk.id = f"{source_id}-chunk-{i}"

        # 生成嵌入
        contents = [c.content for c in chunks]
        embeddings = await self._embedding.embed_batch(contents)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # 存储
        await self._store.add(chunks)

        logger.info(f"索引 {len(chunks)} 个文档块，来源: {source_id}")
        return len(chunks)

    async def index_file(
        self,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """索引文件.

        Args:
            file_path: 文件路径
            metadata: 元数据

        Returns:
            索引的块数量
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件
        content = file_path.read_text(encoding="utf-8")

        # 准备元数据
        file_metadata = dict(metadata) if metadata else {}
        file_metadata["file_path"] = str(file_path)
        file_metadata["file_name"] = file_path.name
        file_metadata["file_type"] = file_path.suffix

        return await self.index_text(
            text=content,
            source_id=str(file_path),
            metadata=file_metadata,
        )

    async def index_knowledge_entry(
        self,
        entry: Any,  # KnowledgeEntry
    ) -> int:
        """索引知识条目.

        Args:
            entry: 知识条目

        Returns:
            索引的块数量
        """
        # 构建索引文本
        parts = [
            f"标题: {entry.title}",
            f"类型: {entry.type.value}",
            f"描述: {entry.description}",
        ]

        if entry.problem:
            parts.append(f"问题: {entry.problem}")
        if entry.root_cause:
            parts.append(f"根本原因: {entry.root_cause}")
        if entry.solution:
            parts.append(f"解决方案: {entry.solution}")
        if entry.prevention:
            parts.append(f"预防措施: {entry.prevention}")

        text = "\n".join(parts)

        metadata = {
            "knowledge_id": entry.id,
            "knowledge_type": entry.type.value,
            "tags": entry.tags,
            "project_type": entry.project_type,
        }

        return await self.index_text(
            text=text,
            source_id=entry.id,
            metadata=metadata,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
        filter_metadata: dict[str, Any] | None = None,
    ) -> RAGContext:
        """检索相关文档.

        Args:
            query: 查询文本
            top_k: 返回数量
            min_score: 最小相关度
            filter_metadata: 元数据过滤

        Returns:
            RAG 上下文
        """
        # 生成查询嵌入
        query_embedding = await self._embedding.embed(query)

        # 搜索
        results = await self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # 过滤低分结果
        filtered_results = [r for r in results if r.score >= min_score]

        # 构建格式化上下文
        formatted = self._format_context(query, filtered_results)

        return RAGContext(
            query=query,
            retrieved_docs=filtered_results,
            formatted_context=formatted,
            metadata={
                "top_k": top_k,
                "min_score": min_score,
                "total_retrieved": len(results),
                "filtered_count": len(filtered_results),
            },
        )

    async def retrieve_for_task(
        self,
        task_description: str,
        task_type: str = "",
        project_type: str = "",
        top_k: int = 5,
    ) -> RAGContext:
        """为任务检索相关知识.

        Args:
            task_description: 任务描述
            task_type: 任务类型
            project_type: 项目类型
            top_k: 返回数量

        Returns:
            RAG 上下文
        """
        # 构建增强查询
        query_parts = [task_description]
        if task_type:
            query_parts.append(f"任务类型: {task_type}")
        if project_type:
            query_parts.append(f"项目类型: {project_type}")

        query = " ".join(query_parts)

        # 元数据过滤
        filter_metadata = {}
        if project_type:
            filter_metadata["project_type"] = [project_type, ""]

        return await self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata if filter_metadata else None,
        )

    async def find_similar_bugs(
        self,
        error_message: str,
        stack_trace: str = "",
        top_k: int = 5,
    ) -> RAGContext:
        """查找相似的 Bug.

        Args:
            error_message: 错误消息
            stack_trace: 堆栈跟踪
            top_k: 返回数量

        Returns:
            RAG 上下文
        """
        query = f"错误: {error_message}"
        if stack_trace:
            # 提取关键信息
            lines = stack_trace.split("\n")[:5]
            query += f"\n堆栈: {' '.join(lines)}"

        return await self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata={"knowledge_type": "bug_fix"},
        )

    def _format_context(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:
        """格式化上下文文本."""
        if not results:
            return ""

        lines = ["相关知识：", ""]

        for i, result in enumerate(results, 1):
            doc = result.document
            score_pct = int(result.score * 100)

            lines.append(f"【{i}】相关度: {score_pct}%")

            # 添加元数据
            if doc.metadata.get("knowledge_type"):
                lines.append(f"类型: {doc.metadata['knowledge_type']}")

            # 添加内容（截断）
            content = doc.content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    async def delete_by_source(self, source_id: str) -> None:
        """删除来源的所有文档.

        Args:
            source_id: 来源 ID
        """
        # 查找所有相关文档
        doc_ids = []
        for i in range(1000):  # 假设最多 1000 个块
            doc_id = f"{source_id}-chunk-{i}"
            doc = await self._store.get(doc_id)
            if doc:
                doc_ids.append(doc_id)
            else:
                break

        if doc_ids:
            await self._store.delete(doc_ids)
            logger.info(f"删除 {len(doc_ids)} 个文档，来源: {source_id}")

    def save(self) -> None:
        """保存索引."""
        if hasattr(self._store, "save"):
            self._store.save()


class RAGPromptBuilder:
    """RAG 提示词构建器."""

    @staticmethod
    def build_task_prompt(
        task_description: str,
        rag_context: RAGContext,
        additional_context: str = "",
    ) -> str:
        """构建任务提示词.

        Args:
            task_description: 任务描述
            rag_context: RAG 上下文
            additional_context: 额外上下文

        Returns:
            完整提示词
        """
        parts = []

        # 知识库上下文
        if rag_context.has_context:
            parts.append("# 相关知识（来自知识库）")
            parts.append("")
            parts.append(rag_context.formatted_context)
            parts.append("")
            parts.append("---")
            parts.append("")

        # 额外上下文
        if additional_context:
            parts.append("# 项目上下文")
            parts.append("")
            parts.append(additional_context)
            parts.append("")
            parts.append("---")
            parts.append("")

        # 任务描述
        parts.append("# 当前任务")
        parts.append("")
        parts.append(task_description)

        return "\n".join(parts)

    @staticmethod
    def build_bugfix_prompt(
        error_message: str,
        stack_trace: str,
        rag_context: RAGContext,
        code_context: str = "",
    ) -> str:
        """构建 Bug 修复提示词.

        Args:
            error_message: 错误消息
            stack_trace: 堆栈跟踪
            rag_context: RAG 上下文
            code_context: 代码上下文

        Returns:
            完整提示词
        """
        parts = []

        # 类似问题
        if rag_context.has_context:
            parts.append("# 类似问题的历史解决方案")
            parts.append("")
            parts.append(rag_context.formatted_context)
            parts.append("")
            parts.append("---")
            parts.append("")

        # 当前错误
        parts.append("# 当前错误")
        parts.append("")
        parts.append(f"**错误消息：**")
        parts.append(f"```")
        parts.append(error_message)
        parts.append("```")
        parts.append("")

        if stack_trace:
            parts.append("**堆栈跟踪：**")
            parts.append("```")
            parts.append(stack_trace)
            parts.append("```")
            parts.append("")

        # 代码上下文
        if code_context:
            parts.append("# 相关代码")
            parts.append("")
            parts.append(code_context)
            parts.append("")

        # 任务要求
        parts.append("---")
        parts.append("")
        parts.append("请分析错误原因，参考历史解决方案，提供修复代码。")

        return "\n".join(parts)


async def create_rag_engine(
    persist_dir: Path | None = None,
    embedding_type: str = "local",
    store_type: str = "memory",
    **kwargs,
) -> RAGEngine:
    """创建 RAG 引擎.

    Args:
        persist_dir: 持久化目录
        embedding_type: 嵌入类型
        store_type: 存储类型
        **kwargs: 其他参数

    Returns:
        RAG 引擎实例
    """
    embedding = create_embedding_provider(embedding_type, **kwargs)

    store_path = persist_dir / "vectors.json" if persist_dir else None
    store = create_vector_store(
        store_type,
        dimension=embedding.dimension,
        persist_path=store_path,
    )

    return RAGEngine(
        vector_store=store,
        embedding_provider=embedding,
        persist_dir=persist_dir,
    )
