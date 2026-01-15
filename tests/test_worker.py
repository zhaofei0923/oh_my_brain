"""Worker模块测试."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from oh_my_brain.schemas.config import WorkerConfig
from oh_my_brain.schemas.task import Task, TaskResult, TaskStatus, TaskType
from oh_my_brain.worker.base import WorkerBase
from oh_my_brain.worker.brain_client import BrainClient


class MockWorker(WorkerBase):
    """测试用Worker实现."""

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.executed_tasks = []

    async def execute_task(self, task: Task) -> TaskResult:
        """模拟执行任务."""
        self.executed_tasks.append(task)
        return TaskResult(
            task_id=task.id,
            success=True,
            output="Task completed",
        )


class TestBrainClient:
    """测试Brain客户端."""

    @pytest.fixture
    def config(self):
        """创建配置."""
        return WorkerConfig(
            brain_address="tcp://127.0.0.1:5555",
            worker_id="test-worker",
        )

    @pytest.fixture
    def client(self, config):
        """创建客户端."""
        return BrainClient(
            brain_address=config.brain_address,
            worker_id=config.worker_id,
        )

    def test_client_initialization(self, client):
        """测试客户端初始化."""
        assert client._worker_id == "test-worker"
        assert client._brain_address == "tcp://127.0.0.1:5555"
        assert not client.is_connected
        assert not client.is_reconnecting

    def test_request_id_generation(self, client):
        """测试请求ID生成."""
        id1 = client._generate_request_id()
        id2 = client._generate_request_id()

        assert id1 != id2
        assert id1.startswith("test-worker-req-")

    @pytest.mark.asyncio
    async def test_reconnect_state(self, client):
        """测试重连状态管理."""
        # 初始状态
        assert not client._reconnecting
        assert client._reconnect_count == 0


class TestWorkerBase:
    """测试Worker基类."""

    @pytest.fixture
    def config(self):
        """创建配置."""
        return WorkerConfig(
            brain_address="tcp://127.0.0.1:5555",
            worker_id="test-worker",
            heartbeat_interval_seconds=1,
        )

    @pytest.fixture
    def worker(self, config):
        """创建Worker."""
        return MockWorker(config)

    def test_worker_initialization(self, worker):
        """测试Worker初始化."""
        assert worker._worker_id == "test-worker"
        assert not worker.is_running
        assert worker.current_task is None

    def test_set_capabilities(self, worker):
        """测试设置能力."""
        worker.set_capabilities(["python", "javascript"])
        assert worker._capabilities == ["python", "javascript"]

    def test_worker_id_property(self, worker):
        """测试worker_id属性."""
        assert worker.worker_id == "test-worker"

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
    async def test_execute_task(self, worker, sample_task):
        """测试任务执行."""
        result = await worker.execute_task(sample_task)

        assert result.success
        assert result.task_id == sample_task.id
        assert sample_task in worker.executed_tasks


class TestWorkerReconnection:
    """测试Worker重连机制."""

    @pytest.fixture
    def client(self):
        """创建客户端."""
        return BrainClient(
            brain_address="tcp://127.0.0.1:5555",
            worker_id="test-worker",
            max_reconnect_attempts=3,
            reconnect_interval=0.1,
        )

    def test_reconnect_config(self, client):
        """测试重连配置."""
        assert client._max_reconnect_attempts == 3
        assert client._reconnect_interval == 0.1

    @pytest.mark.asyncio
    async def test_ensure_connected_raises_when_not_connected(self, client):
        """测试未连接时抛出异常."""
        # 模拟重连失败
        with patch.object(client, 'reconnect', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = False

            with pytest.raises(ConnectionError):
                await client._ensure_connected()


class TestLLMClient:
    """测试LLM客户端."""

    def test_model_prefix_mapping(self):
        """测试模型前缀映射."""
        from oh_my_brain.worker.llm_client import LLMClient

        assert "gpt-" in LLMClient.MODEL_PREFIXES
        assert "claude-" in LLMClient.MODEL_PREFIXES
        assert "deepseek-" in LLMClient.MODEL_PREFIXES

    def test_model_configs(self):
        """测试模型配置."""
        from oh_my_brain.worker.llm_client import LLMClient

        assert "deepseek" in LLMClient.MODEL_CONFIGS
        assert "minimax" in LLMClient.MODEL_CONFIGS

        deepseek_config = LLMClient.MODEL_CONFIGS["deepseek"]
        assert "base_url" in deepseek_config
        assert "env_key" in deepseek_config
