"""知识库与 Worker 集成.

提供任务执行时的知识增强功能。
"""

import logging
from pathlib import Path
from typing import Any

from oh_my_brain.knowledge import (
    BugFixEntry,
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeType,
    RAGContext,
    RAGEngine,
    RAGPromptBuilder,
    create_rag_engine,
)

logger = logging.getLogger(__name__)


class KnowledgeEnhancedWorker:
    """知识增强型 Worker 混入类.

    为 Worker 提供知识库查询和自动学习能力。
    """

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        enable_rag: bool = True,
        enable_learning: bool = True,
        api_key: str | None = None,
    ):
        """初始化.

        Args:
            knowledge_dir: 知识库目录
            enable_rag: 是否启用 RAG
            enable_learning: 是否启用自动学习
            api_key: API Key（用于嵌入）
        """
        self._knowledge_dir = knowledge_dir or Path.home() / ".oh_my_brain" / "knowledge"
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)

        self._enable_rag = enable_rag
        self._enable_learning = enable_learning
        self._api_key = api_key

        # 初始化组件
        self._knowledge_base: KnowledgeBase | None = None
        self._rag_engine: RAGEngine | None = None

        self._initialized = False

    async def initialize(self) -> None:
        """异步初始化."""
        if self._initialized:
            return

        # 加载知识库
        kb_path = self._knowledge_dir / "knowledge_base.json"
        self._knowledge_base = KnowledgeBase(storage_path=kb_path)

        # 初始化 RAG
        if self._enable_rag:
            embedding_type = "minimax" if self._api_key else "local"
            self._rag_engine = await create_rag_engine(
                persist_dir=self._knowledge_dir,
                embedding_type=embedding_type,
                api_key=self._api_key,
            )

            # 索引现有知识
            await self._index_knowledge_base()

        self._initialized = True
        logger.info("知识增强组件初始化完成")

    async def _index_knowledge_base(self) -> None:
        """索引知识库内容."""
        if not self._rag_engine or not self._knowledge_base:
            return

        count = 0
        for entry in self._knowledge_base._entries.values():
            try:
                await self._rag_engine.index_knowledge_entry(entry)
                count += 1
            except Exception as e:
                logger.warning(f"索引知识条目失败: {entry.id}, {e}")

        logger.info(f"索引 {count} 个知识条目")

    async def enhance_task_context(
        self,
        task_description: str,
        task_type: str = "",
        project_type: str = "",
        additional_context: str = "",
    ) -> str:
        """增强任务上下文.

        使用知识库检索相关信息，构建增强的任务提示。

        Args:
            task_description: 任务描述
            task_type: 任务类型
            project_type: 项目类型
            additional_context: 额外上下文

        Returns:
            增强后的提示词
        """
        if not self._initialized:
            await self.initialize()

        if not self._rag_engine:
            return task_description

        # 检索相关知识
        rag_context = await self._rag_engine.retrieve_for_task(
            task_description=task_description,
            task_type=task_type,
            project_type=project_type,
            top_k=5,
        )

        # 构建增强提示
        enhanced_prompt = RAGPromptBuilder.build_task_prompt(
            task_description=task_description,
            rag_context=rag_context,
            additional_context=additional_context,
        )

        logger.debug(f"任务上下文增强: 检索到 {len(rag_context.retrieved_docs)} 条相关知识")
        return enhanced_prompt

    async def enhance_bugfix_context(
        self,
        error_message: str,
        stack_trace: str = "",
        code_context: str = "",
    ) -> str:
        """增强 Bug 修复上下文.

        Args:
            error_message: 错误消息
            stack_trace: 堆栈跟踪
            code_context: 代码上下文

        Returns:
            增强后的提示词
        """
        if not self._initialized:
            await self.initialize()

        if not self._rag_engine:
            return f"错误: {error_message}\n堆栈: {stack_trace}"

        # 查找相似 Bug
        rag_context = await self._rag_engine.find_similar_bugs(
            error_message=error_message,
            stack_trace=stack_trace,
            top_k=5,
        )

        # 构建增强提示
        enhanced_prompt = RAGPromptBuilder.build_bugfix_prompt(
            error_message=error_message,
            stack_trace=stack_trace,
            rag_context=rag_context,
            code_context=code_context,
        )

        logger.debug(f"Bug 修复上下文增强: 找到 {len(rag_context.retrieved_docs)} 个类似问题")
        return enhanced_prompt

    async def learn_from_task_result(
        self,
        task_id: str,
        task_type: str,
        task_description: str,
        result: dict[str, Any],
        project_type: str = "",
    ) -> KnowledgeEntry | None:
        """从任务结果中学习.

        Args:
            task_id: 任务 ID
            task_type: 任务类型
            task_description: 任务描述
            result: 任务结果
            project_type: 项目类型

        Returns:
            创建的知识条目
        """
        if not self._enable_learning or not self._initialized:
            return None

        if not self._knowledge_base:
            await self.initialize()

        # 提取知识
        entry = self._knowledge_base.extract_from_task_result(
            task_id=task_id,
            task_type=task_type,
            result=result,
            project_type=project_type,
        )

        if entry:
            # 保存知识库
            self._knowledge_base.save()

            # 索引新知识
            if self._rag_engine:
                await self._rag_engine.index_knowledge_entry(entry)
                self._rag_engine.save()

            logger.info(f"从任务 {task_id} 学习到新知识: {entry.title}")

        return entry

    async def record_bug_fix(
        self,
        task_id: str,
        error_message: str,
        root_cause: str,
        solution: str,
        code_before: str = "",
        code_after: str = "",
        tags: list[str] | None = None,
        project_type: str = "",
    ) -> BugFixEntry:
        """记录 Bug 修复经验.

        Args:
            task_id: 任务 ID
            error_message: 错误消息
            root_cause: 根本原因
            solution: 解决方案
            code_before: 修复前代码
            code_after: 修复后代码
            tags: 标签
            project_type: 项目类型

        Returns:
            创建的知识条目
        """
        if not self._initialized:
            await self.initialize()

        entry = BugFixEntry(
            id=f"kb-bugfix-{task_id}",
            title=f"Bug 修复: {error_message[:50]}",
            description=f"修复了 '{error_message}' 错误",
            tags=tags or [],
            project_type=project_type,
            problem=error_message,
            root_cause=root_cause,
            solution=solution,
            code_before=code_before,
            code_after=code_after,
            error_message=error_message,
            source=task_id,
        )

        self._knowledge_base.add(entry)
        self._knowledge_base.save()

        if self._rag_engine:
            await self._rag_engine.index_knowledge_entry(entry)
            self._rag_engine.save()

        logger.info(f"记录 Bug 修复经验: {entry.id}")
        return entry

    async def record_best_practice(
        self,
        title: str,
        description: str,
        example_code: str = "",
        tags: list[str] | None = None,
        project_type: str = "",
    ) -> KnowledgeEntry:
        """记录最佳实践.

        Args:
            title: 标题
            description: 描述
            example_code: 示例代码
            tags: 标签
            project_type: 项目类型

        Returns:
            创建的知识条目
        """
        if not self._initialized:
            await self.initialize()

        import uuid
        entry = KnowledgeEntry(
            id=f"kb-bp-{uuid.uuid4().hex[:8]}",
            type=KnowledgeType.BEST_PRACTICE,
            title=title,
            description=description,
            tags=tags or [],
            project_type=project_type,
            solution=example_code,
        )

        self._knowledge_base.add(entry)
        self._knowledge_base.save()

        if self._rag_engine:
            await self._rag_engine.index_knowledge_entry(entry)
            self._rag_engine.save()

        logger.info(f"记录最佳实践: {entry.id}")
        return entry

    def get_knowledge_stats(self) -> dict[str, Any]:
        """获取知识库统计信息."""
        stats = {
            "initialized": self._initialized,
            "rag_enabled": self._enable_rag,
            "learning_enabled": self._enable_learning,
        }

        if self._knowledge_base:
            stats["knowledge_base"] = self._knowledge_base.get_stats()

        if self._rag_engine:
            stats["indexed_documents"] = self._rag_engine.document_count

        return stats

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 10,
        knowledge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """搜索知识库.

        Args:
            query: 搜索查询
            top_k: 返回数量
            knowledge_type: 知识类型过滤

        Returns:
            搜索结果
        """
        if not self._initialized:
            await self.initialize()

        results = []

        # 使用知识库搜索
        if self._knowledge_base:
            types = None
            if knowledge_type:
                try:
                    types = [KnowledgeType(knowledge_type)]
                except ValueError:
                    pass

            entries = self._knowledge_base.search(
                query=query,
                types=types,
                limit=top_k,
            )

            for entry in entries:
                results.append({
                    "id": entry.id,
                    "type": entry.type.value,
                    "title": entry.title,
                    "description": entry.description[:200],
                    "tags": entry.tags,
                })

        return results


def create_knowledge_enhanced_worker(
    knowledge_dir: Path | None = None,
    **kwargs,
) -> KnowledgeEnhancedWorker:
    """创建知识增强 Worker.

    Args:
        knowledge_dir: 知识库目录
        **kwargs: 其他参数

    Returns:
        KnowledgeEnhancedWorker 实例
    """
    return KnowledgeEnhancedWorker(
        knowledge_dir=knowledge_dir,
        **kwargs,
    )
