"""上下文管理器.

集中管理所有Worker的上下文，使用Redis存储。
"""

import json
import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class ContextManager:
    """上下文管理器.

    负责：
    1. 存储/获取Worker上下文（消息历史、笔记）
    2. Token计算和预算管理
    3. 自动摘要压缩
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "omb:",
        token_limit: int = 90000,
        summary_threshold: float = 0.8,
    ):
        self._redis_url = redis_url
        self._prefix = prefix
        self._token_limit = token_limit
        self._summary_threshold = summary_threshold
        self._redis: Any = None  # redis.asyncio.Redis

        # Token计算器
        try:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None
            logger.warning("tiktoken not available, using approximate token count")

    async def connect(self) -> None:
        """连接Redis."""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=False,
            )
            await self._redis.ping()
            logger.info(f"Connected to Redis: {self._redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # 降级到内存存储
            self._redis = None
            self._memory_store: dict[str, dict] = {}
            logger.warning("Using in-memory context store (data will be lost on restart)")

    async def disconnect(self) -> None:
        """断开Redis连接."""
        if self._redis:
            await self._redis.close()

    def _key(self, task_id: str, suffix: str = "") -> str:
        """生成Redis键."""
        return f"{self._prefix}context:{task_id}{':' + suffix if suffix else ''}"

    async def get_context(self, task_id: str) -> dict[str, Any]:
        """获取任务上下文.

        Args:
            task_id: 任务ID

        Returns:
            上下文字典，包含 messages, notes, system_prompt 等
        """
        if self._redis:
            data = await self._redis.get(self._key(task_id))
            if data:
                return json.loads(data)
        else:
            return self._memory_store.get(task_id, {}).copy()

        return {
            "messages": [],
            "notes": [],
            "system_prompt": "",
            "token_count": 0,
        }

    async def update_context(
        self,
        task_id: str,
        context: dict[str, Any],
        merge: bool = True,
    ) -> None:
        """更新任务上下文.

        Args:
            task_id: 任务ID
            context: 上下文数据
            merge: 是否合并（True）还是覆盖（False）
        """
        if merge:
            existing = await self.get_context(task_id)
            # 合并消息
            if "messages" in context:
                existing["messages"] = context["messages"]
            if "notes" in context:
                existing["notes"] = existing.get("notes", []) + context["notes"]
            if "system_prompt" in context:
                existing["system_prompt"] = context["system_prompt"]
            context = existing

        # 计算token
        context["token_count"] = self._count_tokens(context)

        # 检查是否需要压缩
        if context["token_count"] > self._token_limit * self._summary_threshold:
            context = await self._compress_context(context)

        # 存储
        if self._redis:
            await self._redis.set(
                self._key(task_id),
                json.dumps(context, ensure_ascii=False),
            )
        else:
            self._memory_store[task_id] = context

    async def delete_context(self, task_id: str) -> None:
        """删除任务上下文."""
        if self._redis:
            await self._redis.delete(self._key(task_id))
        else:
            self._memory_store.pop(task_id, None)

    async def store_note(self, task_id: str, note: dict[str, Any]) -> None:
        """存储笔记.

        Args:
            task_id: 任务ID
            note: 笔记内容 {"category": "...", "content": "..."}
        """
        context = await self.get_context(task_id)
        notes = context.get("notes", [])
        notes.append(note)
        await self.update_context(task_id, {"notes": notes}, merge=True)

    async def get_notes(self, task_id: str, category: str | None = None) -> list[dict[str, Any]]:
        """获取笔记.

        Args:
            task_id: 任务ID
            category: 分类过滤（可选）

        Returns:
            笔记列表
        """
        context = await self.get_context(task_id)
        notes = context.get("notes", [])
        if category:
            notes = [n for n in notes if n.get("category") == category]
        return notes

    def _count_tokens(self, context: dict[str, Any]) -> int:
        """计算上下文的token数量."""
        total = 0

        # 计算消息token
        for msg in context.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._count_text_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total += self._count_text_tokens(block["text"])

        # 计算笔记token
        for note in context.get("notes", []):
            total += self._count_text_tokens(note.get("content", ""))

        # 计算系统提示词token
        total += self._count_text_tokens(context.get("system_prompt", ""))

        return total

    def _count_text_tokens(self, text: str) -> int:
        """计算文本的token数量."""
        if self._encoding:
            return len(self._encoding.encode(text))
        else:
            # 粗略估算：中文约1.5字符/token，英文约4字符/token
            return len(text) // 3

    async def _compress_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """压缩上下文.

        策略：
        1. 保留最近N条消息
        2. 对旧消息生成摘要
        3. 合并同类笔记
        """
        messages = context.get("messages", [])

        if len(messages) <= 10:
            return context

        # 保留最近10条消息
        recent_messages = messages[-10:]

        # 对旧消息生成简要摘要（这里简化处理，实际应调用LLM）
        old_messages = messages[:-10]
        summary_text = self._create_simple_summary(old_messages)

        # 创建摘要消息
        summary_message = {
            "role": "system",
            "content": f"[对话历史摘要]\n{summary_text}",
        }

        context["messages"] = [summary_message] + recent_messages
        context["token_count"] = self._count_tokens(context)

        logger.info(f"Context compressed: {len(messages)} -> {len(context['messages'])} messages")

        return context

    def _create_simple_summary(self, messages: list[dict]) -> str:
        """创建简单摘要.

        注意：生产环境应使用LLM生成更好的摘要。
        """
        summary_parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str) and len(content) > 100:
                content = content[:100] + "..."

            summary_parts.append(f"[{role}]: {content}")

        # 限制摘要长度
        summary = "\n".join(summary_parts)
        if len(summary) > 1000:
            summary = summary[:1000] + "\n..."

        return summary

    def get_token_budget(self, task_id: str, context: dict[str, Any]) -> int:
        """获取剩余token预算."""
        used = context.get("token_count", 0)
        return max(0, self._token_limit - used)
