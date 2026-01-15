"""Protocol测试."""

import pytest

from oh_my_brain.protocol.messages import (
    WorkerRegisterMessage,
    HeartbeatMessage,
    TaskAssignMessage,
    TaskResultMessage,
    ContextRequest,
    ContextUpdate,
    SafetyCheckRequest,
    SafetyCheckResponse,
)
from oh_my_brain.schemas.task import TaskStatus


class TestMessages:
    """测试Protocol消息."""

    def test_worker_register_message(self):
        """测试Worker注册消息."""
        msg = WorkerRegisterMessage(
            worker_id="worker-1",
            capabilities=["python", "javascript"],
            max_concurrent_tasks=2,
        )

        assert msg.type == "worker_register"
        assert msg.worker_id == "worker-1"
        assert "python" in msg.capabilities

        # 测试序列化
        json_str = msg.model_dump_json()
        assert "worker_register" in json_str

    def test_heartbeat_message(self):
        """测试心跳消息."""
        msg = HeartbeatMessage(
            worker_id="worker-1",
            current_task_id="task-123",
        )

        assert msg.type == "heartbeat"
        assert msg.current_task_id == "task-123"

    def test_task_assign_message(self):
        """测试任务分配消息."""
        msg = TaskAssignMessage(
            task_id="task-1",
            task={
                "id": "task-1",
                "name": "Test Task",
                "type": "feature",
                "description": "Test",
                "requirements": "Do something",
            },
        )

        assert msg.type == "task_assign"
        assert msg.task_id == "task-1"

    def test_task_result_message(self):
        """测试任务结果消息."""
        msg = TaskResultMessage(
            worker_id="worker-1",
            task_id="task-1",
            result={
                "task_id": "task-1",
                "status": "completed",
                "output": "Done",
            },
        )

        assert msg.type == "task_result"
        assert msg.result["status"] == "completed"

    def test_context_request(self):
        """测试上下文请求."""
        msg = ContextRequest(
            worker_id="worker-1",
            request_id="req-1",
            keys=["project_root", "git_branch"],
        )

        assert msg.type == "context_request"
        assert len(msg.keys) == 2

    def test_context_update(self):
        """测试上下文更新."""
        msg = ContextUpdate(
            worker_id="worker-1",
            key="last_file",
            value="src/main.py",
        )

        assert msg.type == "context_update"
        assert msg.key == "last_file"

    def test_safety_check_request(self):
        """测试安全检查请求."""
        msg = SafetyCheckRequest(
            worker_id="worker-1",
            request_id="req-1",
            command_type="bash",
            command="rm -rf /tmp/test",
            args={},
        )

        assert msg.type == "safety_check_request"
        assert msg.command_type == "bash"

    def test_safety_check_response(self):
        """测试安全检查响应."""
        msg = SafetyCheckResponse(
            request_id="req-1",
            approved=True,
            reason="",
        )

        assert msg.type == "safety_check_response"
        assert msg.approved is True
