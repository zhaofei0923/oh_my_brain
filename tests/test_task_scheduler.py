"""任务调度器测试."""

import pytest
from datetime import datetime

from oh_my_brain.brain.task_scheduler import TaskScheduler
from oh_my_brain.brain.task_persistence import MemoryTaskPersistence
from oh_my_brain.schemas.task import Task, TaskStatus, TaskType
from oh_my_brain.schemas.dev_doc import DevDoc, Module, SubTask, ProjectInfo, TechStack


class TestTaskScheduler:
    """测试任务调度器."""

    @pytest.fixture
    def scheduler(self):
        """创建调度器."""
        return TaskScheduler()

    @pytest.fixture
    def sample_dev_doc(self):
        """创建示例开发文档."""
        return DevDoc(
            project=ProjectInfo(
                name="Test Project",
                description="A test project",
                tech_stack=TechStack(language="Python"),
            ),
            modules=[
                Module(
                    id="mod-1",
                    name="Module 1",
                    description="First module",
                    acceptance_criteria="Tests pass",
                    sub_tasks=[
                        SubTask(
                            id="task-1",
                            name="Task 1",
                            type="feature",
                            description="First task",
                            requirements="Do something",
                        ),
                        SubTask(
                            id="task-2",
                            name="Task 2",
                            type="feature",
                            description="Second task",
                            requirements="Do something else",
                        ),
                    ],
                ),
                Module(
                    id="mod-2",
                    name="Module 2",
                    description="Second module",
                    acceptance_criteria="Tests pass",
                    dependencies=["mod-1"],
                    sub_tasks=[
                        SubTask(
                            id="task-3",
                            name="Task 3",
                            type="feature",
                            description="Third task",
                            requirements="Depends on module 1",
                        ),
                    ],
                ),
            ],
        )

    def test_load_from_dev_doc(self, scheduler, sample_dev_doc):
        """测试从开发文档加载任务."""
        tasks = scheduler.load_from_dev_doc(sample_dev_doc)

        assert len(tasks) == 3
        assert all(isinstance(t, Task) for t in tasks)

    def test_task_dependencies(self, scheduler, sample_dev_doc):
        """测试任务依赖关系."""
        scheduler.load_from_dev_doc(sample_dev_doc)
        all_tasks = scheduler.get_all_tasks()

        # 模块2的任务应该依赖模块1的所有任务
        mod2_tasks = [t for t in all_tasks if t.module_id == "mod-2"]
        assert len(mod2_tasks) == 1

        mod2_task = mod2_tasks[0]
        mod1_task_ids = [t.id for t in all_tasks if t.module_id == "mod-1"]

        # 模块2的任务应该依赖模块1的任务
        for dep in mod1_task_ids:
            assert dep in mod2_task.dependencies

    def test_get_next_task(self, scheduler, sample_dev_doc):
        """测试获取下一个任务."""
        scheduler.load_from_dev_doc(sample_dev_doc)

        # 第一个任务应该是模块1的第一个任务
        task = scheduler.get_next_task("worker-1")
        assert task is not None
        assert task.module_id == "mod-1"
        assert task.status == TaskStatus.ASSIGNED

    def test_task_status_flow(self, scheduler, sample_dev_doc):
        """测试任务状态流转."""
        scheduler.load_from_dev_doc(sample_dev_doc)

        # 获取任务
        task = scheduler.get_next_task("worker-1")
        assert task.status == TaskStatus.ASSIGNED

        # 标记开始
        scheduler.mark_started(task.id, "worker-1", "gpt-4", "feature/task-1")
        updated_task = scheduler.get_task(task.id)
        assert updated_task.status == TaskStatus.RUNNING

        # 标记完成
        scheduler.mark_completed(task.id)
        updated_task = scheduler.get_task(task.id)
        assert updated_task.status == TaskStatus.COMPLETED

    def test_requeue_task(self, scheduler, sample_dev_doc):
        """测试任务重新入队."""
        scheduler.load_from_dev_doc(sample_dev_doc)

        task = scheduler.get_next_task("worker-1")
        task_id = task.id

        # 重新入队
        scheduler.requeue_task(task_id)
        updated_task = scheduler.get_task(task_id)
        assert updated_task.status == TaskStatus.PENDING

    def test_get_status(self, scheduler, sample_dev_doc):
        """测试获取调度器状态."""
        scheduler.load_from_dev_doc(sample_dev_doc)

        status = scheduler.get_status()
        assert status["total_tasks"] == 3
        assert status["pending_queue_size"] > 0

    def test_dependency_blocking(self, scheduler, sample_dev_doc):
        """测试依赖阻塞."""
        scheduler.load_from_dev_doc(sample_dev_doc)

        # 获取所有可用任务
        available_tasks = []
        while True:
            task = scheduler.get_next_task(f"worker-{len(available_tasks)}")
            if task is None:
                break
            available_tasks.append(task)

        # 模块2的任务不应该在模块1完成前可用
        mod2_tasks = [t for t in available_tasks if t.module_id == "mod-2"]
        assert len(mod2_tasks) == 0

    def test_dependency_unblocking(self, scheduler, sample_dev_doc):
        """测试依赖解除."""
        scheduler.load_from_dev_doc(sample_dev_doc)

        # 完成模块1的所有任务
        while True:
            task = scheduler.get_next_task("worker-1")
            if task is None or task.module_id != "mod-1":
                break
            scheduler.mark_completed(task.id)

        # 现在模块2的任务应该可用
        task = scheduler.get_next_task("worker-1")
        assert task is not None
        assert task.module_id == "mod-2"


class TestTaskPersistence:
    """测试任务持久化."""

    @pytest.fixture
    def persistence(self):
        """创建内存持久化."""
        return MemoryTaskPersistence()

    @pytest.fixture
    def sample_task(self):
        """创建示例任务."""
        return Task(
            id="task-001",
            module_id="mod-1",
            sub_task_id="subtask-1",
            name="Test Task",
            type=TaskType.CODING,
            description="A test task",
            requirements="Test requirements",
        )

    @pytest.mark.asyncio
    async def test_save_and_load_task(self, persistence, sample_task):
        """测试保存和加载任务."""
        await persistence.save_task(sample_task)

        loaded = await persistence.load_task(sample_task.id)
        assert loaded is not None
        assert loaded.id == sample_task.id
        assert loaded.name == sample_task.name

    @pytest.mark.asyncio
    async def test_load_all_tasks(self, persistence, sample_task):
        """测试加载所有任务."""
        # 保存多个任务
        await persistence.save_task(sample_task)

        task2 = Task(
            id="task-002",
            module_id="mod-1",
            sub_task_id="subtask-2",
            name="Test Task 2",
            type=TaskType.CODING,
            description="Another test task",
            requirements="Test requirements",
        )
        await persistence.save_task(task2)

        tasks = await persistence.load_all_tasks()
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_delete_task(self, persistence, sample_task):
        """测试删除任务."""
        await persistence.save_task(sample_task)
        await persistence.delete_task(sample_task.id)

        loaded = await persistence.load_task(sample_task.id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, persistence, sample_task):
        """测试保存和加载检查点."""
        await persistence.save_task(sample_task)

        checkpoint = {
            "step": 5,
            "progress": 0.5,
            "data": {"key": "value"},
        }
        await persistence.save_checkpoint(sample_task.id, checkpoint)

        loaded = await persistence.load_checkpoint(sample_task.id)
        assert loaded is not None
        assert loaded["step"] == 5
        assert loaded["progress"] == 0.5
        assert "saved_at" in loaded

    @pytest.mark.asyncio
    async def test_clear_all(self, persistence, sample_task):
        """测试清除所有数据."""
        await persistence.save_task(sample_task)
        await persistence.save_checkpoint(sample_task.id, {"step": 1})

        await persistence.clear_all()

        tasks = await persistence.load_all_tasks()
        assert len(tasks) == 0
