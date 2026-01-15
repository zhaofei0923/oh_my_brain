"""任务调度器.

负责任务队列管理、DAG依赖调度、负载均衡。
"""

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
    """

    def __init__(self):
        # 任务存储
        self._tasks: dict[str, Task] = {}  # task_id -> Task
        self._module_tasks: dict[str, list[str]] = defaultdict(list)  # module_id -> [task_ids]

        # 队列
        self._pending_queue: list[str] = []  # 待分配任务ID列表
        self._running_tasks: dict[str, str] = {}  # task_id -> worker_id

        # 统计
        self._completed_count = 0
        self._failed_count = 0

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
        module_deps = dev_doc.get_task_dag()

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
            tid for tid, task in self._tasks.items()
            if task.status == TaskStatus.COMPLETED
        }

        # 找出所有就绪的任务
        ready_tasks = []
        for task_id, task in self._tasks.items():
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

        return task

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
            logger.info(f"Task {task_id} started by {worker_id}")

    def mark_completed(self, task_id: str) -> None:
        """标记任务完成."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_completed()
            self._running_tasks.pop(task_id, None)
            self._completed_count += 1
            logger.info(f"Task {task_id} completed")

            # 更新队列（可能有新任务就绪）
            self._update_pending_queue()

    def mark_failed(self, task_id: str, error: str) -> None:
        """标记任务失败."""
        task = self._tasks.get(task_id)
        if task:
            task.mark_failed(error)
            self._running_tasks.pop(task_id, None)
            self._failed_count += 1
            logger.warning(f"Task {task_id} failed: {error}")

    def requeue_task(self, task_id: str) -> None:
        """重新入队任务（用于Worker超时等情况）."""
        task = self._tasks.get(task_id)
        if task and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
            task.status = TaskStatus.PENDING
            task.assigned_worker = None
            self._running_tasks.pop(task_id, None)
            self._update_pending_queue()
            logger.info(f"Task {task_id} requeued")

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
        status_counts = defaultdict(int)
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
