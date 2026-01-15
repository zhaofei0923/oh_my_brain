"""任务调度器.

负责任务队列管理、DAG依赖调度、负载均衡。
支持任务持久化和断点续传。
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

from oh_my_brain.schemas.dev_doc import DevDoc
from oh_my_brain.schemas.dev_doc import TaskType as DocTaskType
from oh_my_brain.schemas.task import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)


class TaskScheduler:
    """任务调度器.

    功能：
    1. 维护任务队列（按优先级和依赖关系）
    2. 分配任务给Worker（负载均衡）
    3. 跟踪任务状态
    4. 任务持久化和断点续传
    """

    def __init__(self, persistence=None):
        """初始化调度器.

        Args:
            persistence: 持久化后端（可选）
        """
        # 任务存储
        self._tasks: dict[str, Task] = {}  # task_id -> Task
        self._module_tasks: dict[str, list[str]] = defaultdict(list)  # module_id -> [task_ids]

        # 队列
        self._pending_queue: list[str] = []  # 待分配任务ID列表
        self._running_tasks: dict[str, str] = {}  # task_id -> worker_id

        # 统计
        self._completed_count = 0
        self._failed_count = 0

        # 持久化
        self._persistence = persistence

    async def set_persistence(self, persistence) -> None:
        """设置持久化后端并加载已保存的任务."""
        from oh_my_brain.brain.task_persistence import TaskPersistence

        self._persistence = persistence

        if self._persistence:
            await self._load_persisted_tasks()

    async def _load_persisted_tasks(self) -> None:
        """加载已持久化的任务."""
        if not self._persistence:
            return

        try:
            tasks = await self._persistence.load_all_tasks()
            for task in tasks:
                self._tasks[task.id] = task
                self._module_tasks[task.module_id].append(task.id)

                # 更新统计
                if task.status == TaskStatus.COMPLETED:
                    self._completed_count += 1
                elif task.status == TaskStatus.FAILED:
                    self._failed_count += 1

            self._update_pending_queue()
            logger.info(f"Loaded {len(tasks)} persisted tasks")

        except Exception as e:
            logger.error(f"Failed to load persisted tasks: {e}")

    async def _persist_task(self, task: Task) -> None:
        """持久化单个任务."""
        if self._persistence:
            try:
                await self._persistence.save_task(task)
            except Exception as e:
                logger.error(f"Failed to persist task {task.id}: {e}")

    async def save_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> None:
        """保存任务检查点.

        Args:
            task_id: 任务ID
            checkpoint: 检查点数据
        """
        task = self._tasks.get(task_id)
        if task:
            task.checkpoint = checkpoint
            if self._persistence:
                try:
                    await self._persistence.save_checkpoint(task_id, checkpoint)
                except Exception as e:
                    logger.error(f"Failed to save checkpoint for task {task_id}: {e}")

    async def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """加载任务检查点.

        Args:
            task_id: 任务ID

        Returns:
            检查点数据
        """
        if self._persistence:
            try:
                return await self._persistence.load_checkpoint(task_id)
            except Exception as e:
                logger.error(f"Failed to load checkpoint for task {task_id}: {e}")

        task = self._tasks.get(task_id)
        return task.checkpoint if task else None

    async def resume_incomplete_tasks(self) -> list[Task]:
        """恢复未完成的任务.

        Returns:
            需要恢复的任务列表
        """
        incomplete = []
        for task in self._tasks.values():
            if task.status in [TaskStatus.RUNNING, TaskStatus.ASSIGNED]:
                # 这些任务需要重新分配
                task.status = TaskStatus.PENDING
                task.assigned_worker = None
                await self._persist_task(task)
                incomplete.append(task)

        self._update_pending_queue()
        logger.info(f"Found {len(incomplete)} tasks to resume")
        return incomplete

    def load_from_dev_doc(self, dev_doc: DevDoc) -> list[Task]:
        """从开发文档加载任务.

        Args:
            dev_doc: 开发文档

        Returns:
            创建的任务列表
        """
        tasks = []
        task_counter = 0

        # 获取模块依赖关系
        dev_doc.get_task_dag()

        for module in dev_doc.modules:
            # 计算模块内任务的依赖
            prev_task_id: str | None = None

            for sub_task in module.sub_tasks:
                task_counter += 1
                task_id = f"task-{task_counter:04d}"

                # 确定任务类型
                task_type = self._map_task_type(sub_task.type)

                # 构建依赖列表
                dependencies = []
                # 模块间依赖：等待依赖模块的所有任务完成
                for dep_module_id in module.dependencies:
                    if dep_module_id in self._module_tasks:
                        dependencies.extend(self._module_tasks[dep_module_id])
                # 模块内依赖：串行执行
                if prev_task_id:
                    dependencies.append(prev_task_id)

                task = Task(
                    id=task_id,
                    module_id=module.id,
                    sub_task_id=sub_task.id,
                    name=sub_task.name,
                    type=task_type,
                    description=sub_task.description,
                    requirements=sub_task.requirements,
                    files_involved=sub_task.files_involved,
                    priority=module.priority,
                    dependencies=dependencies,
                    estimated_minutes=sub_task.estimated_minutes,
                    assigned_model=sub_task.preferred_model,
                )

                self._tasks[task_id] = task
                self._module_tasks[module.id].append(task_id)
                tasks.append(task)
                prev_task_id = task_id

        # 初始化待分配队列
        self._update_pending_queue()

        logger.info(f"Loaded {len(tasks)} tasks from dev doc")
        return tasks

    def _map_task_type(self, doc_type: DocTaskType) -> TaskType:
        """映射开发文档任务类型到调度任务类型."""
        mapping = {
            DocTaskType.FEATURE: TaskType.CODING,
            DocTaskType.BUGFIX: TaskType.CODING,
            DocTaskType.REFACTOR: TaskType.REFACTOR,
            DocTaskType.TEST: TaskType.TESTING,
            DocTaskType.DOCS: TaskType.DOCS,
        }
        return mapping.get(doc_type, TaskType.CODING)

    def _update_pending_queue(self) -> None:
        """更新待分配队列."""
        completed_tasks = {
            tid for tid, task in self._tasks.items() if task.status == TaskStatus.COMPLETED
        }

        # 找出所有就绪的任务
        ready_tasks = []
        for _task_id, task in self._tasks.items():
            if task.status == TaskStatus.PENDING and task.is_ready(completed_tasks):
                ready_tasks.append(task)

        # 按优先级排序
        ready_tasks.sort(key=lambda t: (t.priority, t.created_at))

        self._pending_queue = [t.id for t in ready_tasks]

    def get_next_task(self, worker_id: str) -> Task | None:
        """获取下一个待分配任务.

        Args:
            worker_id: 请求任务的Worker ID

        Returns:
            任务对象，如果没有可用任务则返回None
        """
        self._update_pending_queue()

        if not self._pending_queue:
            return None

        task_id = self._pending_queue.pop(0)
        task = self._tasks.get(task_id)

        if task:
            task.status = TaskStatus.ASSIGNED
            self._running_tasks[task_id] = worker_id
            # 异步持久化（不阻塞）
            asyncio.create_task(self._persist_task(task))

        return task

    async def mark_started_async(
        self,
        task_id: str,
        worker_id: str,
        model: str,
        branch: str,
    ) -> None:
        """标记任务开始执行（异步版本）."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_started(worker_id, model, branch)
            self._running_tasks[task_id] = worker_id
            await self._persist_task(task)
            logger.info(f"Task {task_id} started by {worker_id}")

    def mark_started(
        self,
        task_id: str,
        worker_id: str,
        model: str,
        branch: str,
    ) -> None:
        """标记任务开始执行."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_started(worker_id, model, branch)
            self._running_tasks[task_id] = worker_id
            # 异步持久化
            asyncio.create_task(self._persist_task(task))
            logger.info(f"Task {task_id} started by {worker_id}")

    async def mark_completed_async(self, task_id: str) -> None:
        """标记任务完成（异步版本）."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_completed()
            self._running_tasks.pop(task_id, None)
            self._completed_count += 1
            await self._persist_task(task)
            logger.info(f"Task {task_id} completed")
            self._update_pending_queue()

    def mark_completed(self, task_id: str) -> None:
        """标记任务完成."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_completed()
            self._running_tasks.pop(task_id, None)
            self._completed_count += 1
            # 异步持久化
            asyncio.create_task(self._persist_task(task))
            logger.info(f"Task {task_id} completed")

            # 更新队列（可能有新任务就绪）
            self._update_pending_queue()

    async def mark_failed_async(self, task_id: str, error: str) -> None:
        """标记任务失败（异步版本）."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_failed(error)
            self._running_tasks.pop(task_id, None)
            self._failed_count += 1
            await self._persist_task(task)
            logger.warning(f"Task {task_id} failed: {error}")

    def mark_failed(self, task_id: str, error: str) -> None:
        """标记任务失败."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_failed(error)
            self._running_tasks.pop(task_id, None)
            self._failed_count += 1
            # 异步持久化
            asyncio.create_task(self._persist_task(task))
            logger.warning(f"Task {task_id} failed: {error}")

    async def requeue_task_async(self, task_id: str) -> None:
        """重新入队任务（异步版本）."""
        task = self._tasks.get(task_id)
        if task and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
            task.status = TaskStatus.PENDING
            task.assigned_worker = None
            self._running_tasks.pop(task_id, None)
            await self._persist_task(task)
            self._update_pending_queue()
            logger.info(f"Task {task_id} requeued")

    def requeue_task(self, task_id: str) -> None:
        """重新入队任务（用于Worker超时等情况）."""
        task = self._tasks.get(task_id)
        if task and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
            task.status = TaskStatus.PENDING
            task.assigned_worker = None
            self._running_tasks.pop(task_id, None)
            # 异步持久化
            asyncio.create_task(self._persist_task(task))
            self._update_pending_queue()
            logger.info(f"Task {task_id} requeued")

    async def update_progress_async(self, task_id: str, progress: float, checkpoint: dict[str, Any] | None = None) -> None:
        """更新任务进度（异步版本，支持检查点）."""
        task = self._tasks.get(task_id)
        if task:
            task.progress = min(1.0, max(0.0, progress))
            if checkpoint:
                await self.save_checkpoint(task_id, checkpoint)
            else:
                await self._persist_task(task)

    def update_progress(self, task_id: str, progress: float) -> None:
        """更新任务进度."""
        task = self._tasks.get(task_id)
        if task:
            task.progress = min(1.0, max(0.0, progress))

    def get_task(self, task_id: str) -> Task | None:
        """获取任务."""
        return self._tasks.get(task_id)

    def get_module_tasks(self, module_id: str) -> list[Task]:
        """获取模块的所有任务."""
        task_ids = self._module_tasks.get(module_id, [])
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def get_status(self) -> dict[str, Any]:
        """获取调度器状态."""
        status_counts: dict[str, int] = defaultdict(int)
        for task in self._tasks.values():
            status_counts[task.status.value] += 1

        return {
            "total_tasks": len(self._tasks),
            "pending_queue_size": len(self._pending_queue),
            "running_tasks": len(self._running_tasks),
            "completed": self._completed_count,
            "failed": self._failed_count,
            "by_status": dict(status_counts),
        }

    def get_all_tasks(self) -> list[Task]:
        """获取所有任务."""
        return list(self._tasks.values())

    def get_running_tasks(self) -> dict[str, str]:
        """获取正在运行的任务（task_id -> worker_id）."""
        return self._running_tasks.copy()

    def add_task(
        self,
        module_id: str,
        name: str,
        description: str,
        task_type: str = "coding",
        requirements: list[str] | None = None,
        files_involved: list[str] | None = None,
        priority: int = 0,
        depends_on: list[str] | None = None,
    ) -> Task:
        """手动添加任务.

        Args:
            module_id: 模块ID
            name: 任务名称
            description: 任务描述
            task_type: 任务类型
            requirements: 需求列表
            files_involved: 涉及的文件
            priority: 优先级
            depends_on: 依赖的任务ID列表

        Returns:
            创建的任务
        """
        import uuid

        # 映射任务类型
        type_mapping = {
            "coding": TaskType.CODING,
            "development": TaskType.CODING,
            "testing": TaskType.TESTING,
            "refactor": TaskType.REFACTOR,
            "docs": TaskType.DOCS,
            "review": TaskType.REVIEW,
        }
        task_type_enum = type_mapping.get(task_type.lower(), TaskType.CODING)

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        task = Task(
            id=task_id,
            module_id=module_id,
            name=name,
            type=task_type_enum,
            description=description,
            requirements=requirements or [],
            files_involved=files_involved or [],
            priority=priority,
            dependencies=depends_on or [],
        )

        self._tasks[task_id] = task
        self._module_tasks[module_id].append(task_id)

        # 异步持久化
        asyncio.create_task(self._persist_task(task))

        self._update_pending_queue()
        logger.info(f"Task {task_id} added: {name}")

        return task

    async def add_task_async(
        self,
        module_id: str,
        name: str,
        description: str,
        task_type: str = "coding",
        requirements: list[str] | None = None,
        files_involved: list[str] | None = None,
        priority: int = 0,
        depends_on: list[str] | None = None,
    ) -> Task:
        """手动添加任务（异步版本）."""
        task = self.add_task(
            module_id=module_id,
            name=name,
            description=description,
            task_type=task_type,
            requirements=requirements,
            files_involved=files_involved,
            priority=priority,
            depends_on=depends_on,
        )
        await self._persist_task(task)
        return task

    def mark_cancelled(self, task_id: str) -> None:
        """取消任务.

        Args:
            task_id: 任务ID
        """
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            self._running_tasks.pop(task_id, None)
            # 异步持久化
            asyncio.create_task(self._persist_task(task))
            self._update_pending_queue()
            logger.info(f"Task {task_id} cancelled")

    async def mark_cancelled_async(self, task_id: str) -> None:
        """取消任务（异步版本）."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            self._running_tasks.pop(task_id, None)
            await self._persist_task(task)
            self._update_pending_queue()
            logger.info(f"Task {task_id} cancelled")
