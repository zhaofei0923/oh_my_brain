"""Protocol测试."""

from oh_my_brain.protocol.messages import (
    BrainMessage,
    ContextGetRequest,
    ContextUpdateRequest,
    HeartbeatMessage,
    MessageType,
    SafetyCheckRequest,
    SafetyCheckResponse,
    TaskAssignMessage,
    TaskResultMessage,
    WorkerRegisterMessage,
)


class TestMessages:
    """测试Protocol消息."""

    def test_worker_register_message(self):
        """测试Worker注册消息."""
        payload = WorkerRegisterMessage.Payload(
            worker_id="worker-1",
            hostname="localhost",
            platform="linux",
            capabilities=["python", "javascript"],
            max_concurrent_tasks=2,
        )
        msg = WorkerRegisterMessage(
            msg_id="msg-001",
            sender="worker-1",
            payload=payload,
        )

        assert msg.msg_type == MessageType.WORKER_REGISTER
        assert msg.payload.worker_id == "worker-1"
        assert "python" in msg.payload.capabilities

        # 测试序列化
        json_str = msg.model_dump_json()
        assert "worker_register" in json_str

    def test_heartbeat_message(self):
        """测试心跳消息."""
        payload = HeartbeatMessage.Payload(
            worker_id="worker-1",
            status="busy",
            current_task_id="task-123",
        )
        msg = HeartbeatMessage(
            msg_id="msg-002",
            sender="worker-1",
            payload=payload,
        )

        assert msg.msg_type == MessageType.HEARTBEAT
        assert msg.payload.current_task_id == "task-123"

    def test_task_assign_message(self):
        """测试任务分配消息."""
        payload = TaskAssignMessage.Payload(
            task_id="task-1",
            module_id="mod-1",
            name="Test Task",
            type="feature",
            description="Test",
            requirements="Do something",
            git_branch="agent/task-1",
            model_name="claude-3-opus",
        )
        msg = TaskAssignMessage(
            msg_id="msg-003",
            sender="brain",
            payload=payload,
        )

        assert msg.msg_type == MessageType.TASK_ASSIGN
        assert msg.payload.task_id == "task-1"

    def test_task_result_message(self):
        """测试任务结果消息."""
        payload = TaskResultMessage.Payload(
            task_id="task-1",
            success=True,
            output="Done",
        )
        msg = TaskResultMessage(
            msg_id="msg-004",
            sender="worker-1",
            payload=payload,
        )

        assert msg.msg_type == MessageType.TASK_RESULT
        assert msg.payload.success is True

    def test_context_get_request(self):
        """测试上下文请求."""
        payload = ContextGetRequest.Payload(
            worker_id="worker-1",
            task_id="task-1",
        )
        msg = ContextGetRequest(
            msg_id="msg-005",
            sender="worker-1",
            payload=payload,
        )

        assert msg.msg_type == MessageType.CONTEXT_GET

    def test_context_update_request(self):
        """测试上下文更新."""
        payload = ContextUpdateRequest.Payload(
            worker_id="worker-1",
            task_id="task-1",
            messages=[{"role": "user", "content": "hello"}],
        )
        msg = ContextUpdateRequest(
            msg_id="msg-006",
            sender="worker-1",
            payload=payload,
        )

        assert msg.msg_type == MessageType.CONTEXT_UPDATE

    def test_safety_check_request(self):
        """测试安全检查请求."""
        payload = SafetyCheckRequest.Payload(
            worker_id="worker-1",
            task_id="task-1",
            command_type="bash",
            command="rm -rf /tmp/test",
        )
        msg = SafetyCheckRequest(
            msg_id="msg-007",
            sender="worker-1",
            payload=payload,
        )

        assert msg.msg_type == MessageType.SAFETY_CHECK
        assert msg.payload.command_type == "bash"

    def test_safety_check_response(self):
        """测试安全检查响应."""
        payload = SafetyCheckResponse.Payload(
            approved=True,
            reason="",
        )
        msg = SafetyCheckResponse(
            msg_id="msg-008",
            sender="brain",
            payload=payload,
        )

        assert msg.msg_type == MessageType.SAFETY_RESPONSE
        assert msg.payload.approved is True

    def test_message_serialization(self):
        """测试消息序列化."""
        payload = HeartbeatMessage.Payload(
            worker_id="worker-1",
            status="idle",
        )
        msg = HeartbeatMessage(
            msg_id="msg-009",
            sender="worker-1",
            payload=payload,
        )

        # 序列化
        data = msg.to_bytes()
        assert isinstance(data, bytes)

        # 反序列化
        restored = BrainMessage.from_bytes(data)
        assert restored.msg_id == msg.msg_id
        assert restored.sender == msg.sender
