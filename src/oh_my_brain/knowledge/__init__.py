"""知识库模块.

用于积累和管理开发过程中的知识，包括：
1. Bug 修复经验
2. 最佳实践
3. 常见问题解决方案
4. 代码模式和反模式
5. RAG 检索增强生成
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# 导出 RAG 组件
from oh_my_brain.knowledge.rag import (
    RAGContext,
    RAGEngine,
    RAGPromptBuilder,
    create_rag_engine,
)
from oh_my_brain.knowledge.vector_store import (
    Document,
    EmbeddingProvider,
    InMemoryVectorStore,
    LocalEmbedding,
    MiniMaxEmbedding,
    SearchResult,
    VectorStore,
    create_embedding_provider,
    create_vector_store,
)

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """知识类型."""

    BUG_FIX = "bug_fix"             # Bug 修复经验
    BEST_PRACTICE = "best_practice"  # 最佳实践
    PATTERN = "pattern"              # 代码模式
    ANTI_PATTERN = "anti_pattern"    # 反模式（应避免）
    TIP = "tip"                      # 开发技巧
    LESSON = "lesson"                # 经验教训
    FAQ = "faq"                      # 常见问题


class KnowledgeEntry(BaseModel):
    """知识条目."""

    id: str
    type: KnowledgeType
    title: str
    description: str
    tags: list[str] = Field(default_factory=list)

    # 上下文信息
    project_type: str = ""           # 相关项目类型
    tech_stack: list[str] = Field(default_factory=list)
    files_involved: list[str] = Field(default_factory=list)

    # 问题与解决方案
    problem: str = ""                # 问题描述
    root_cause: str = ""             # 根本原因
    solution: str = ""               # 解决方案
    code_before: str = ""            # 修复前代码
    code_after: str = ""             # 修复后代码

    # 预防措施
    prevention: str = ""             # 如何预防
    related_ids: list[str] = Field(default_factory=list)

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    author: str = ""
    source: str = ""                 # 来源（任务 ID、PR 等）
    confidence: float = 1.0          # 置信度 0-1

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEntry":
        """从字典创建."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class BugFixEntry(KnowledgeEntry):
    """Bug 修复知识条目."""

    type: KnowledgeType = KnowledgeType.BUG_FIX

    # Bug 特定字段
    error_message: str = ""          # 错误消息
    stack_trace: str = ""            # 堆栈跟踪
    reproduction_steps: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    severity: str = "medium"         # low, medium, high, critical
    fix_time_minutes: int = 0        # 修复耗时


class KnowledgeBase:
    """知识库.

    功能：
    1. 存储和检索知识条目
    2. 基于相似度搜索
    3. 标签和类型过滤
    4. 自动从任务结果中提取知识
    """

    def __init__(self, storage_path: Path | None = None):
        """初始化知识库.

        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path
        self._entries: dict[str, KnowledgeEntry] = {}
        self._tag_index: dict[str, set[str]] = {}
        self._type_index: dict[KnowledgeType, set[str]] = {}

        if storage_path and storage_path.exists():
            self._load()

    @property
    def entry_count(self) -> int:
        """获取条目数量."""
        return len(self._entries)

    def add(self, entry: KnowledgeEntry) -> None:
        """添加知识条目.

        Args:
            entry: 知识条目
        """
        self._entries[entry.id] = entry
        self._index_entry(entry)
        logger.info(f"添加知识条目: {entry.id} - {entry.title}")

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        """获取知识条目.

        Args:
            entry_id: 条目 ID

        Returns:
            知识条目
        """
        return self._entries.get(entry_id)

    def update(self, entry_id: str, updates: dict[str, Any]) -> None:
        """更新知识条目.

        Args:
            entry_id: 条目 ID
            updates: 更新内容
        """
        entry = self._entries.get(entry_id)
        if not entry:
            raise ValueError(f"条目不存在: {entry_id}")

        # 移除旧索引
        self._unindex_entry(entry)

        # 应用更新
        for key, value in updates.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        entry.updated_at = datetime.now()

        # 重建索引
        self._index_entry(entry)
        logger.info(f"更新知识条目: {entry_id}")

    def remove(self, entry_id: str) -> KnowledgeEntry | None:
        """删除知识条目.

        Args:
            entry_id: 条目 ID

        Returns:
            被删除的条目
        """
        entry = self._entries.pop(entry_id, None)
        if entry:
            self._unindex_entry(entry)
            logger.info(f"删除知识条目: {entry_id}")
        return entry

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        types: list[KnowledgeType] | None = None,
        project_type: str = "",
        limit: int = 20,
    ) -> list[KnowledgeEntry]:
        """搜索知识条目.

        Args:
            query: 搜索关键词
            tags: 标签过滤
            types: 类型过滤
            project_type: 项目类型过滤
            limit: 最大返回数量

        Returns:
            匹配的条目列表
        """
        candidates = set(self._entries.keys())

        # 标签过滤
        if tags:
            tag_matches = set()
            for tag in tags:
                tag_matches.update(self._tag_index.get(tag.lower(), set()))
            candidates &= tag_matches

        # 类型过滤
        if types:
            type_matches = set()
            for t in types:
                type_matches.update(self._type_index.get(t, set()))
            candidates &= type_matches

        # 获取候选条目
        results = [self._entries[eid] for eid in candidates]

        # 项目类型过滤
        if project_type:
            results = [e for e in results if e.project_type == project_type or not e.project_type]

        # 关键词搜索
        if query:
            query_lower = query.lower()
            scored_results = []
            for entry in results:
                score = self._compute_relevance(entry, query_lower)
                if score > 0:
                    scored_results.append((score, entry))

            scored_results.sort(key=lambda x: x[0], reverse=True)
            results = [e for _, e in scored_results[:limit]]
        else:
            # 按更新时间排序
            results.sort(key=lambda e: e.updated_at, reverse=True)
            results = results[:limit]

        return results

    def find_similar_bugs(
        self,
        error_message: str,
        stack_trace: str = "",
        limit: int = 5,
    ) -> list[BugFixEntry]:
        """查找相似的 Bug 修复经验.

        Args:
            error_message: 错误消息
            stack_trace: 堆栈跟踪
            limit: 最大返回数量

        Returns:
            相似的 Bug 修复条目
        """
        bug_ids = self._type_index.get(KnowledgeType.BUG_FIX, set())
        bugs = [self._entries[eid] for eid in bug_ids if isinstance(self._entries.get(eid), BugFixEntry)]

        if not bugs:
            return []

        # 计算相似度
        scored_bugs = []
        error_lower = error_message.lower()
        stack_lower = stack_trace.lower()

        for bug in bugs:
            score = 0.0

            # 错误消息相似度
            if bug.error_message:
                bug_error_lower = bug.error_message.lower()
                if error_lower in bug_error_lower or bug_error_lower in error_lower:
                    score += 5.0
                else:
                    # 检查关键词重叠
                    error_words = set(error_lower.split())
                    bug_words = set(bug_error_lower.split())
                    overlap = len(error_words & bug_words)
                    if overlap > 0:
                        score += overlap * 0.5

            # 堆栈跟踪相似度
            if stack_trace and bug.stack_trace:
                bug_stack_lower = bug.stack_trace.lower()
                # 检查函数名重叠
                import re
                error_funcs = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', stack_lower))
                bug_funcs = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', bug_stack_lower))
                func_overlap = len(error_funcs & bug_funcs)
                score += func_overlap * 0.3

            if score > 0:
                scored_bugs.append((score, bug))

        scored_bugs.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored_bugs[:limit]]

    def get_best_practices(
        self,
        project_type: str = "",
        tech_stack: list[str] | None = None,
    ) -> list[KnowledgeEntry]:
        """获取最佳实践.

        Args:
            project_type: 项目类型
            tech_stack: 技术栈

        Returns:
            最佳实践列表
        """
        return self.search(
            types=[KnowledgeType.BEST_PRACTICE],
            project_type=project_type,
            limit=50,
        )

    def get_anti_patterns(
        self,
        project_type: str = "",
    ) -> list[KnowledgeEntry]:
        """获取反模式（应避免的做法）.

        Args:
            project_type: 项目类型

        Returns:
            反模式列表
        """
        return self.search(
            types=[KnowledgeType.ANTI_PATTERN],
            project_type=project_type,
            limit=50,
        )

    # ========== 知识提取 ==========

    def extract_from_task_result(
        self,
        task_id: str,
        task_type: str,
        result: dict[str, Any],
        project_type: str = "",
    ) -> KnowledgeEntry | None:
        """从任务结果中提取知识.

        Args:
            task_id: 任务 ID
            task_type: 任务类型
            result: 任务结果
            project_type: 项目类型

        Returns:
            提取的知识条目（如果有）
        """
        # Bug 修复任务
        if task_type == "bugfix" and result.get("success"):
            entry = BugFixEntry(
                id=f"kb-{task_id}",
                title=result.get("summary", "Bug 修复"),
                description=result.get("description", ""),
                tags=result.get("tags", []),
                project_type=project_type,
                files_involved=result.get("files_modified", []),
                problem=result.get("problem", ""),
                root_cause=result.get("root_cause", ""),
                solution=result.get("solution", ""),
                code_before=result.get("code_before", ""),
                code_after=result.get("code_after", ""),
                error_message=result.get("error_message", ""),
                source=task_id,
            )
            self.add(entry)
            return entry

        # 重构任务 - 提取模式
        elif task_type == "refactor" and result.get("success"):
            entry = KnowledgeEntry(
                id=f"kb-{task_id}",
                type=KnowledgeType.PATTERN,
                title=result.get("summary", "代码重构"),
                description=result.get("description", ""),
                tags=["refactor"] + result.get("tags", []),
                project_type=project_type,
                files_involved=result.get("files_modified", []),
                problem=result.get("before_state", ""),
                solution=result.get("after_state", ""),
                code_before=result.get("code_before", ""),
                code_after=result.get("code_after", ""),
                source=task_id,
            )
            self.add(entry)
            return entry

        return None

    # ========== 索引管理 ==========

    def _index_entry(self, entry: KnowledgeEntry) -> None:
        """为条目建立索引."""
        # 标签索引
        for tag in entry.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(entry.id)

        # 类型索引
        if entry.type not in self._type_index:
            self._type_index[entry.type] = set()
        self._type_index[entry.type].add(entry.id)

    def _unindex_entry(self, entry: KnowledgeEntry) -> None:
        """移除条目索引."""
        # 标签索引
        for tag in entry.tags:
            tag_lower = tag.lower()
            if tag_lower in self._tag_index:
                self._tag_index[tag_lower].discard(entry.id)

        # 类型索引
        if entry.type in self._type_index:
            self._type_index[entry.type].discard(entry.id)

    def _compute_relevance(self, entry: KnowledgeEntry, query: str) -> float:
        """计算相关性分数."""
        score = 0.0
        query_words = set(query.split())

        # 标题匹配
        title_lower = entry.title.lower()
        if query in title_lower:
            score += 10.0
        else:
            title_words = set(title_lower.split())
            overlap = len(query_words & title_words)
            score += overlap * 2.0

        # 描述匹配
        desc_lower = entry.description.lower()
        if query in desc_lower:
            score += 5.0
        else:
            desc_words = set(desc_lower.split())
            overlap = len(query_words & desc_words)
            score += overlap * 1.0

        # 问题匹配
        problem_lower = entry.problem.lower()
        if query in problem_lower:
            score += 3.0

        # 解决方案匹配
        solution_lower = entry.solution.lower()
        if query in solution_lower:
            score += 2.0

        # 标签匹配
        for tag in entry.tags:
            if query in tag.lower():
                score += 3.0

        return score

    # ========== 持久化 ==========

    def save(self, path: Path | None = None) -> None:
        """保存知识库.

        Args:
            path: 保存路径
        """
        save_path = path or self._storage_path
        if not save_path:
            raise ValueError("未指定存储路径")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "entries": [e.to_dict() for e in self._entries.values()],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"保存知识库: {save_path}, {len(self._entries)} 条目")

    def _load(self) -> None:
        """加载知识库."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry_type = entry_data.get("type")
                if entry_type == KnowledgeType.BUG_FIX.value:
                    entry = BugFixEntry.from_dict(entry_data)
                else:
                    entry = KnowledgeEntry.from_dict(entry_data)
                self._entries[entry.id] = entry
                self._index_entry(entry)

            logger.info(f"加载知识库: {len(self._entries)} 条目")

        except Exception as e:
            logger.error(f"加载知识库失败: {e}")

    # ========== 统计 ==========

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息."""
        type_counts = {}
        for t in KnowledgeType:
            type_counts[t.value] = len(self._type_index.get(t, set()))

        tag_counts = {
            tag: len(ids)
            for tag, ids in sorted(self._tag_index.items(), key=lambda x: len(x[1]), reverse=True)[:20]
        }

        return {
            "total_entries": len(self._entries),
            "by_type": type_counts,
            "top_tags": tag_counts,
        }

    def export_markdown(self, output_path: Path) -> None:
        """导出为 Markdown 文档.

        Args:
            output_path: 输出路径
        """
        lines = [
            "# 开发知识库",
            "",
            f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*条目总数: {len(self._entries)}*",
            "",
        ]

        # 按类型分组
        for t in KnowledgeType:
            type_ids = self._type_index.get(t, set())
            if not type_ids:
                continue

            type_names = {
                KnowledgeType.BUG_FIX: "Bug 修复经验",
                KnowledgeType.BEST_PRACTICE: "最佳实践",
                KnowledgeType.PATTERN: "代码模式",
                KnowledgeType.ANTI_PATTERN: "反模式",
                KnowledgeType.TIP: "开发技巧",
                KnowledgeType.LESSON: "经验教训",
                KnowledgeType.FAQ: "常见问题",
            }

            lines.append(f"## {type_names.get(t, t.value)}")
            lines.append("")

            for eid in type_ids:
                entry = self._entries[eid]
                lines.append(f"### {entry.title}")
                lines.append("")

                if entry.tags:
                    lines.append(f"**标签:** {', '.join(entry.tags)}")
                    lines.append("")

                if entry.description:
                    lines.append(entry.description)
                    lines.append("")

                if entry.problem:
                    lines.append("**问题:**")
                    lines.append(f"> {entry.problem}")
                    lines.append("")

                if entry.root_cause:
                    lines.append("**根本原因:**")
                    lines.append(f"> {entry.root_cause}")
                    lines.append("")

                if entry.solution:
                    lines.append("**解决方案:**")
                    lines.append(entry.solution)
                    lines.append("")

                if entry.code_before and entry.code_after:
                    lines.append("**修复前:**")
                    lines.append("```python")
                    lines.append(entry.code_before)
                    lines.append("```")
                    lines.append("")
                    lines.append("**修复后:**")
                    lines.append("```python")
                    lines.append(entry.code_after)
                    lines.append("```")
                    lines.append("")

                if entry.prevention:
                    lines.append("**预防措施:**")
                    lines.append(entry.prevention)
                    lines.append("")

                lines.append("---")
                lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"导出知识库: {output_path}")
