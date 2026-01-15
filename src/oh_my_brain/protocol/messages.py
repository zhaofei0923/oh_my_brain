"""消息定义.

定义Brain与Worker之间的通信消息格式。
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """消息类型."""

    # Worker -> Brain
    WORKER_REGISTER = "worker_register"  # Worker注册
    WORKER_UNREGISTER = "worker_unregister"  # Worker注销
    HEARTBEAT = "heartbeat"  # 心跳
    TASK_REQUEST = "task_request"  # 请求任务
    TASK_RESULT = "task_result"  # 任务结果
    CONTEXT_GET = "context_get"  # 获取上下文
    CONTEXT_UPDATE = "context_update"  # 更新上下文
    SAFETY_CHECK = "safety_check"  # 安全检查请求
    CHECKPOINT = "checkpoint"  # 检查点保存

    # Brain -> Worker
    TASK_ASSIGN = "task_assign"  # 分配任务
    TASK_CANCEL = "task_cancel"  # 取消任务
    CONTEXT_RESPONSE = "context_response"  # 上下文响应
    SAFETY_RESPONSE = "safety_response"  # 安全检查响应
    CONFIG_UPDATE = "config_update"  # 配置更新
    SHUTDOWN = "shutdown"  # 关闭命令

    # 通用
    ACK = "ack"  # 确认
    ERROR = "error"  # 错误


class BrainMessage(BaseModel):
    """消息基类."""

    msg_type: MessageType = Field(..., description="消息类型")
    msg_id: str = Field(..., description="消息唯一ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    sender: str = Field(..., description="发送者ID")
    payload: dict[str, Any] = Field(default_factory=dict, description="消息负载")

    model_config = {"extra": "forbid"}

    def to_bytes(self) -> bytes:
        """序列化为字节."""
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "BrainMessage":
        """从字节反序列化."""
        return cls.model_validate_json(data.decode("utf-8"))


# ============ Worker -> Brain 消息 ============


class WorkerRegisterMessage(BrainMessage):
    """Worker注册消息."""

    msg_type: MessageType = MessageType.WORKER_REGISTER

    class Payload(BaseModel):
        worker_id: str
        hostname: str
        platform: str  # linux, windows, darwin
        capabilities: list[str] = Field(default_factory=list)
        max_concurrent_tasks: int = 1

    payload: Payload  # type: ignore[assignment]


class HeartbeatMessage(BrainMessage):
    """心跳消息."""

    msg_type: MessageType = MessageType.HEARTBEAT

    class Payload(BaseModel):
        worker_id: str
        status: str  # idle, busy, paused
        current_task_id: str | None = None
        task_progress: float = 0.0
        memory_mb: float = 0.0
        cpu_percent: float = 0.0

    payload: Payload  # type: ignore[assignment]


class TaskResultMessage(BrainMessage):
    """任务结果消息."""

    msg_type: MessageType = MessageType.TASK_RESULT

    class Payload(BaseModel):
        task_id: str
        success: bool
        output: str = ""
        error: str | None = None
        files_modified: list[str] = Field(default_factory=list)
        git_commits: list[str] = Field(default_factory=list)
        tokens_used: int = 0
        duration_seconds: float = 0.0
        model_used: str = ""
        context_update: dict[str, Any] | None = None

    payload: Payload  # type: ignore[assignment]


class ContextGetRequest(BrainMessage):
    """获取上下文请求."""

    msg_type: MessageType = MessageType.CONTEXT_GET

    class Payload(BaseModel):
        worker_id: str
        task_id: str

    payload: Payload  # type: ignore[assignment]


class ContextUpdateRequest(BrainMessage):
    """更新上下文请求."""

    msg_type: MessageType = MessageType.CONTEXT_UPDATE

    class Payload(BaseModel):
        worker_id: str
        task_id: str
        messages: list[dict[str, Any]]  # 消息历史
        notes: list[dict[str, Any]] = Field(default_factory=list)  # 笔记

    payload: Payload  # type: ignore[assignment]


class SafetyCheckRequest(BrainMessage):
    """安全检查请求."""

    msg_type: MessageType = MessageType.SAFETY_CHECK

    class Payload(BaseModel):
        worker_id: str
        task_id: str
        command_type: str  # bash, write, edit
        command: str  # 要执行的命令
        args: dict[str, Any] = Field(default_factory=dict)

    payload: Payload  # type: ignore[assignment]


# ============ Brain -> Worker 消息 ============


class TaskAssignMessage(BrainMessage):
    """任务分配消息."""

    msg_type: MessageType = MessageType.TASK_ASSIGN

    class Payload(BaseModel):
        task_id: str
        module_id: str
        name: str
        type: str
        description: str
        requirements: str
        files_involved: list[str] = Field(default_factory=list)
        git_branch: str
        model_name: str
        llm_config: dict[str, Any] = Field(default_factory=dict)  # 模型配置
        context: dict[str, Any] = Field(default_factory=dict)  # 初始上下文

    payload: Payload  # type: ignore[assignment]


class ContextGetResponse(BrainMessage):
    """上下文响应."""

    msg_type: MessageType = MessageType.CONTEXT_RESPONSE

    class Payload(BaseModel):
        task_id: str
        messages: list[dict[str, Any]]  # 消息历史
        notes: list[dict[str, Any]] = Field(default_factory=list)  # 笔记
        system_prompt: str = ""  # 系统提示词
        token_budget: int = 90000  # 可用token预算

    payload: Payload  # type: ignore[assignment]


class SafetyCheckResponse(BrainMessage):
    """安全检查响应."""

    msg_type: MessageType = MessageType.SAFETY_RESPONSE

    class Payload(BaseModel):
        approved: bool
        reason: str = ""
        modified_command: str | None = None  # Brain修改后的命令（可选）

    payload: Payload  # type: ignore[assignment]
