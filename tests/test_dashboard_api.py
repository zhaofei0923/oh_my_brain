"""Dashboard API 测试."""

import pytest

# 只有在 FastAPI 可用时才运行测试
try:
    from fastapi.testclient import TestClient
    from oh_my_brain.api.dashboard import FASTAPI_AVAILABLE, create_dashboard_app

    HAS_FASTAPI = FASTAPI_AVAILABLE
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestDashboardAPI:
    """Dashboard API 测试."""

    @pytest.fixture
    def client(self):
        """创建测试客户端."""
        app = create_dashboard_app()
        return TestClient(app)

    def test_health_check(self, client):
        """测试健康检查."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_status_without_brain(self, client):
        """测试无 Brain 时的状态."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is False
        assert data["workers_count"] == 0

    def test_list_tasks_empty(self, client):
        """测试空任务列表."""
        response = client.get("/api/tasks")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_workers_empty(self, client):
        """测试空 Worker 列表."""
        response = client.get("/api/workers")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_approvals_empty(self, client):
        """测试空审批列表."""
        response = client.get("/api/approvals")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_task_not_found(self, client):
        """测试获取不存在的任务."""
        response = client.get("/api/tasks/nonexistent")
        assert response.status_code == 503  # Brain not available

    def test_get_worker_not_found(self, client):
        """测试获取不存在的 Worker."""
        response = client.get("/api/workers/nonexistent")
        assert response.status_code == 503  # Brain not available

    def test_create_task_without_brain(self, client):
        """测试无 Brain 时创建任务."""
        response = client.post("/api/tasks", json={
            "module_id": "mod1",
            "name": "Test Task",
            "description": "Test description",
        })
        assert response.status_code == 503  # Brain not available

    def test_approve_nonexistent(self, client):
        """测试批准不存在的请求."""
        response = client.post("/api/approvals/nonexistent/approve", json={
            "request_id": "nonexistent",
            "approved": True,
        })
        assert response.status_code == 404

    def test_reject_nonexistent(self, client):
        """测试拒绝不存在的请求."""
        response = client.post("/api/approvals/nonexistent/reject", json={
            "request_id": "nonexistent",
            "approved": False,
        })
        assert response.status_code == 404


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestDashboardWithMockBrain:
    """带模拟 Brain 的 Dashboard 测试."""

    class MockTaskScheduler:
        """模拟任务调度器."""

        def __init__(self):
            self._tasks = {}

        def get_status(self):
            return {
                "pending": 5,
                "running": 2,
                "completed": 10,
            }

        def add_task(self, **kwargs):
            from unittest.mock import MagicMock
            task = MagicMock()
            task.id = "task-001"
            task.name = kwargs.get("name", "Test")
            return task

        def mark_cancelled(self, task_id):
            pass

    class MockBrain:
        """模拟 Brain 服务器."""

        def __init__(self):
            self._task_scheduler = TestDashboardWithMockBrain.MockTaskScheduler()

        def get_status(self):
            return {
                "running": True,
                "workers": {
                    "worker-1": {
                        "worker_id": "worker-1",
                        "hostname": "localhost",
                        "platform": "linux",
                        "status": "idle",
                        "current_task_id": None,
                        "last_heartbeat": "2025-01-15T10:00:00",
                    }
                },
                "tasks": self._task_scheduler.get_status(),
            }

    @pytest.fixture
    def client_with_brain(self):
        """创建带模拟 Brain 的测试客户端."""
        app = create_dashboard_app(self.MockBrain())
        return TestClient(app)

    def test_status_with_brain(self, client_with_brain):
        """测试有 Brain 时的状态."""
        response = client_with_brain.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True
        assert data["workers_count"] == 1

    def test_list_workers_with_brain(self, client_with_brain):
        """测试有 Brain 时的 Worker 列表."""
        response = client_with_brain.get("/api/workers")
        assert response.status_code == 200
        workers = response.json()
        assert len(workers) == 1
        assert workers[0]["worker_id"] == "worker-1"

    def test_get_worker_with_brain(self, client_with_brain):
        """测试获取 Worker."""
        response = client_with_brain.get("/api/workers/worker-1")
        assert response.status_code == 200
        worker = response.json()
        assert worker["worker_id"] == "worker-1"

    def test_create_task_with_brain(self, client_with_brain):
        """测试创建任务."""
        response = client_with_brain.post("/api/tasks", json={
            "module_id": "mod1",
            "name": "Test Task",
            "description": "Test description",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "id" in data
