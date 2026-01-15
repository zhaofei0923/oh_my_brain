"""Brain客户端.

Worker用于与Brain通信的客户端。
"""

import asyncio
import logging
from typing import Any

import zmq
import zmq.asyncio

from oh_my_brain.protocol.messages import (
    BrainMessage,
    ContextRequest,
    ContextUpdate,
    HeartbeatMessage,
    SafetyCheckRequest,
    TaskResultMessage,
    TaskStatusUpdate,
    WorkerRegisterMessage,
)
from oh_my_brain.schemas.task import TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class BrainClient:
    """Brain客户端.

    使用ZeroMQ DEALER socket与Brain通信。
    """

    def __init__(
        self,
        brain_address: str,
        worker_id: str,
    ):
        self._brain_address = brain_address
        self._worker_id = worker_id

        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._connected = False

        # 请求响应追踪
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._request_counter = 0

    @property
    def is_connected(self) -> bool:
        """是否已连接."""
        return self._connected

    async def connect(self) -> None:
        """连接到Brain."""
        if self._connected:
            return

        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)

        # 设置socket identity
        self._socket.setsockopt_string(zmq.IDENTITY, self._worker_id)

        # 设置超时
        self._socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒接收超时
        self._socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5秒发送超时
        self._socket.setsockopt(zmq.LINGER, 0)  # 关闭时不等待

        self._socket.connect(self._brain_address)
        self._connected = True

        logger.info(f"Connected to Brain at {self._brain_address}")

    async def disconnect(self) -> None:
        """断开连接."""
        if not self._connected:
            return

        if self._socket:
            self._socket.close()
            self._socket = None

        if self._context:
            self._context.term()
            self._context = None

        self._connected = False
        logger.info("Disconnected from Brain")

    async def send(self, message: BrainMessage) -> None:
        """发送消息.

        Args:
            message: 消息对象
        """
        if not self._socket:
            raise RuntimeError("Not connected to Brain")

        data = message.model_dump_json().encode()
        await self._socket.send(data)

    async def receive(self, timeout: float | None = None) -> dict[str, Any] | None:
        """接收消息.

        Args:
            timeout: 超时时间（秒），None表示使用默认超时

        Returns:
            消息字典，超时返回None
        """
        if not self._socket:
            return None

        try:
            if timeout is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))

            data = await self._socket.recv()
            import json

            return json.loads(data.decode())

        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Receive error: {e}")
            return None

    async def register(
        self,
        capabilities: list[str] | None = None,
        max_concurrent_tasks: int = 1,
    ) -> bool:
        """注册Worker.

        Args:
            capabilities: 能力列表
            max_concurrent_tasks: 最大并发任务数

        Returns:
            是否成功
        """
        message = WorkerRegisterMessage(
            worker_id=self._worker_id,
            capabilities=capabilities or [],
            max_concurrent_tasks=max_concurrent_tasks,
        )

        await self.send(message)

        # 等待确认
        response = await self.receive(timeout=10)
        if response and response.get("type") == "register_ack":
            logger.info(f"Worker {self._worker_id} registered successfully")
            return True
        else:
            logger.error("Worker registration failed")
            return False

    async def send_heartbeat(
        self,
        current_task_id: str | None = None,
    ) -> None:
        """发送心跳.

        Args:
            current_task_id: 当前任务ID
        """
        message = HeartbeatMessage(
            worker_id=self._worker_id,
            current_task_id=current_task_id,
        )
        await self.send(message)

    async def report_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        """上报任务状态.

        Args:
            task_id: 任务ID
            status: 任务状态
            progress: 进度 (0-1)
            message: 状态消息
        """
        msg = TaskStatusUpdate(
            worker_id=self._worker_id,
            task_id=task_id,
            status=status,
            progress=progress,
            message=message,
        )
        await self.send(msg)

    async def report_task_result(
        self,
        task_id: str,
        result: TaskResult,
    ) -> None:
        """上报任务结果.

        Args:
            task_id: 任务ID
            result: 任务结果
        """
        msg = TaskResultMessage(
            worker_id=self._worker_id,
            task_id=task_id,
            result=result.model_dump(),
        )
        await self.send(msg)

    async def request_context(self, keys: list[str]) -> dict[str, Any]:
        """请求上下文.

        Args:
            keys: 需要的上下文键列表

        Returns:
            上下文数据
        """
        request_id = self._generate_request_id()

        msg = ContextRequest(
            worker_id=self._worker_id,
            request_id=request_id,
            keys=keys,
        )
        await self.send(msg)

        # 等待响应
        response = await self.receive(timeout=30)
        if response and response.get("type") == "context_response":
            return response.get("data", {})
        else:
            logger.warning("Context request timed out or failed")
            return {}

    async def update_context(self, key: str, value: Any) -> None:
        """更新上下文.

        Args:
            key: 上下文键
            value: 上下文值
        """
        msg = ContextUpdate(
            worker_id=self._worker_id,
            key=key,
            value=value,
        )
        await self.send(msg)

    async def request_safety_check(
        self,
        command_type: str,
        command: str,
        args: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """请求安全检查.

        Args:
            command_type: 命令类型
            command: 命令内容
            args: 额外参数

        Returns:
            (是否批准, 原因)
        """
        request_id = self._generate_request_id()

        msg = SafetyCheckRequest(
            worker_id=self._worker_id,
            request_id=request_id,
            command_type=command_type,
            command=command,
            args=args or {},
        )
        await self.send(msg)

        # 等待响应
        response = await self.receive(timeout=30)
        if response and response.get("type") == "safety_check_response":
            return response.get("approved", False), response.get("reason", "")
        else:
            # 超时默认拒绝
            return False, "Safety check timed out"

    def _generate_request_id(self) -> str:
        """生成请求ID."""
        self._request_counter += 1
        return f"{self._worker_id}-req-{self._request_counter}"
