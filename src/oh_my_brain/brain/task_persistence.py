"""任务持久化.

支持将任务状态保存到 Redis 或文件，实现断点续传。
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from oh_my_brain.schemas.task import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskPersistence(ABC):
    """任务持久化抽象基类."""

    @abstractmethod
    async def save_task(self, task: Task) -> None:
        """保存任务."""
        pass

    @abstractmethod
    async def load_task(self, task_id: str) -> Task | None:
        """加载任务."""
        pass

    @abstractmethod
    async def load_all_tasks(self) -> list[Task]:
        """加载所有任务."""
        pass

    @abstractmethod
    async def delete_task(self, task_id: str) -> None:
        """删除任务."""
        pass

    @abstractmethod
    async def save_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> None:
        """保存检查点."""
        pass

    @abstractmethod
    async def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """加载检查点."""
        pass

    @abstractmethod
    async def clear_all(self) -> None:
        """清除所有数据."""
        pass


class RedisTaskPersistence(TaskPersistence):
    """Redis 任务持久化.

    使用 Redis 存储任务状态和检查点。
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "omb:task:",
    ):
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._checkpoint_prefix = f"{key_prefix}checkpoint:"
        self._client = None

    async def _get_client(self):
        """获取 Redis 客户端."""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self._redis_url)
                # 测试连接
                await self._client.ping()
                logger.info(f"Connected to Redis at {self._redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._client

    async def save_task(self, task: Task) -> None:
        """保存任务到 Redis."""
        client = await self._get_client()
        key = f"{self._key_prefix}{task.id}"

        # 序列化任务
        data = task.model_dump_json()
        await client.set(key, data)

        # 添加到任务索引
        await client.sadd(f"{self._key_prefix}index", task.id)

        logger.debug(f"Saved task {task.id} to Redis")

    async def load_task(self, task_id: str) -> Task | None:
        """从 Redis 加载任务."""
        client = await self._get_client()
        key = f"{self._key_prefix}{task_id}"

        data = await client.get(key)
        if data is None:
            return None

        return Task.model_validate_json(data)

    async def load_all_tasks(self) -> list[Task]:
        """加载所有任务."""
        client = await self._get_client()

        # 获取所有任务 ID
        task_ids = await client.smembers(f"{self._key_prefix}index")
        if not task_ids:
            return []

        tasks = []
        for task_id in task_ids:
            task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
            task = await self.load_task(task_id_str)
            if task:
                tasks.append(task)

        logger.info(f"Loaded {len(tasks)} tasks from Redis")
        return tasks

    async def delete_task(self, task_id: str) -> None:
        """删除任务."""
        client = await self._get_client()
        key = f"{self._key_prefix}{task_id}"

        await client.delete(key)
        await client.srem(f"{self._key_prefix}index", task_id)

        # 同时删除检查点
        await client.delete(f"{self._checkpoint_prefix}{task_id}")

    async def save_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> None:
        """保存检查点."""
        client = await self._get_client()
        key = f"{self._checkpoint_prefix}{task_id}"

        # 添加时间戳
        checkpoint["saved_at"] = datetime.now().isoformat()
        data = json.dumps(checkpoint)
        await client.set(key, data)

        # 同时更新任务的 checkpoint 字段
        task = await self.load_task(task_id)
        if task:
            task.checkpoint = checkpoint
            await self.save_task(task)

        logger.debug(f"Saved checkpoint for task {task_id}")

    async def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """加载检查点."""
        client = await self._get_client()
        key = f"{self._checkpoint_prefix}{task_id}"

        data = await client.get(key)
        if data is None:
            return None

        return json.loads(data)

    async def clear_all(self) -> None:
        """清除所有任务数据."""
        client = await self._get_client()

        # 获取所有任务 ID
        task_ids = await client.smembers(f"{self._key_prefix}index")

        # 删除所有任务和检查点
        for task_id in task_ids:
            task_id_str = task_id.decode() if isinstance(task_id, bytes) else task_id
            await client.delete(f"{self._key_prefix}{task_id_str}")
            await client.delete(f"{self._checkpoint_prefix}{task_id_str}")

        # 删除索引
        await client.delete(f"{self._key_prefix}index")

        logger.info("Cleared all tasks from Redis")

    async def get_incomplete_tasks(self) -> list[Task]:
        """获取未完成的任务（用于恢复）."""
        all_tasks = await self.load_all_tasks()

        incomplete = [
            task for task in all_tasks
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
        ]

        return incomplete

    async def close(self) -> None:
        """关闭连接."""
        if self._client:
            await self._client.close()
            self._client = None


class FileTaskPersistence(TaskPersistence):
    """文件任务持久化.

    使用 JSON 文件存储任务状态（适合开发/测试）。
    """

    def __init__(self, data_dir: Path | str = ".oh-my-brain/tasks"):
        self._data_dir = Path(data_dir)
        self._tasks_file = self._data_dir / "tasks.json"
        self._checkpoints_dir = self._data_dir / "checkpoints"

        # 确保目录存在
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _load_tasks_file(self) -> dict[str, dict]:
        """加载任务文件."""
        if not self._tasks_file.exists():
            return {}

        try:
            with open(self._tasks_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load tasks file: {e}")
            return {}

    def _save_tasks_file(self, data: dict[str, dict]) -> None:
        """保存任务文件."""
        with open(self._tasks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    async def save_task(self, task: Task) -> None:
        """保存任务到文件."""
        tasks = self._load_tasks_file()
        tasks[task.id] = json.loads(task.model_dump_json())
        self._save_tasks_file(tasks)
        logger.debug(f"Saved task {task.id} to file")

    async def load_task(self, task_id: str) -> Task | None:
        """从文件加载任务."""
        tasks = self._load_tasks_file()
        if task_id not in tasks:
            return None
        return Task.model_validate(tasks[task_id])

    async def load_all_tasks(self) -> list[Task]:
        """加载所有任务."""
        tasks = self._load_tasks_file()
        return [Task.model_validate(data) for data in tasks.values()]

    async def delete_task(self, task_id: str) -> None:
        """删除任务."""
        tasks = self._load_tasks_file()
        if task_id in tasks:
            del tasks[task_id]
            self._save_tasks_file(tasks)

        # 删除检查点
        checkpoint_file = self._checkpoints_dir / f"{task_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    async def save_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> None:
        """保存检查点."""
        checkpoint["saved_at"] = datetime.now().isoformat()
        checkpoint_file = self._checkpoints_dir / f"{task_id}.json"

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)

        # 同时更新任务
        task = await self.load_task(task_id)
        if task:
            task.checkpoint = checkpoint
            await self.save_task(task)

        logger.debug(f"Saved checkpoint for task {task_id}")

    async def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """加载检查点."""
        checkpoint_file = self._checkpoints_dir / f"{task_id}.json"
        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def clear_all(self) -> None:
        """清除所有数据."""
        if self._tasks_file.exists():
            self._tasks_file.unlink()

        for f in self._checkpoints_dir.glob("*.json"):
            f.unlink()

        logger.info("Cleared all tasks from file storage")

    async def get_incomplete_tasks(self) -> list[Task]:
        """获取未完成的任务."""
        all_tasks = await self.load_all_tasks()
        return [
            task for task in all_tasks
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
        ]


class MemoryTaskPersistence(TaskPersistence):
    """内存任务持久化（用于测试）."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}

    async def save_task(self, task: Task) -> None:
        self._tasks[task.id] = task

    async def load_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    async def load_all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    async def delete_task(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)
        self._checkpoints.pop(task_id, None)

    async def save_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> None:
        checkpoint["saved_at"] = datetime.now().isoformat()
        self._checkpoints[task_id] = checkpoint
        if task_id in self._tasks:
            self._tasks[task_id].checkpoint = checkpoint

    async def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        return self._checkpoints.get(task_id)

    async def clear_all(self) -> None:
        self._tasks.clear()
        self._checkpoints.clear()


def create_persistence(
    backend: str = "file",
    **kwargs,
) -> TaskPersistence:
    """创建持久化后端.

    Args:
        backend: 后端类型 ('redis', 'file', 'memory')
        **kwargs: 后端特定参数

    Returns:
        持久化实例
    """
    if backend == "redis":
        return RedisTaskPersistence(**kwargs)
    elif backend == "file":
        return FileTaskPersistence(**kwargs)
    elif backend == "memory":
        return MemoryTaskPersistence()
    else:
        raise ValueError(f"Unknown persistence backend: {backend}")
