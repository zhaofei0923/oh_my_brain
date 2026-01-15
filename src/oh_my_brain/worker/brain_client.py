"""Brain客户端.

Worker用于与Brain通信的客户端。
支持自动重连和错误恢复。
"""

import asyncio
import logging
from typing import Any

import zmq
import zmq.asyncio

from oh_my_brain.protocol.messages import (
    BrainMessage,
    ContextGetRequest,
    ContextUpdateRequest,
    HeartbeatMessage,
    SafetyCheckRequest,
    TaskResultMessage,
    WorkerRegisterMessage,
)
from oh_my_brain.schemas.task import TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class BrainClient:
    """Brain客户端.

    使用ZeroMQ DEALER socket与Brain通信。
    支持自动重连和错误恢复。
    """

    def __init__(
        self,
        brain_address: str,
        worker_id: str,
        max_reconnect_attempts: int = 10,
        reconnect_interval: float = 2.0,
    ):
        self._brain_address = brain_address
        self._worker_id = worker_id
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_interval = reconnect_interval

        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._connected = False

        # 重连状态
        self._reconnecting = False
        self._reconnect_count = 0

        # 请求响应追踪
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._request_counter = 0

        # 注册信息（用于重连后重新注册）
        self._registered = False
        self._capabilities: list[str] = []
        self._max_concurrent_tasks: int = 1

    @property
    def is_connected(self) -> bool:
        """是否已连接."""
        return self._connected

    @property
    def is_reconnecting(self) -> bool:
        """是否正在重连."""
        return self._reconnecting

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

        # 重连相关设置
        self._socket.setsockopt(zmq.RECONNECT_IVL, 1000)  # 1秒重连间隔
        self._socket.setsockopt(zmq.RECONNECT_IVL_MAX, 10000)  # 最大10秒

        self._socket.connect(self._brain_address)
        self._connected = True
        self._reconnect_count = 0

        logger.info(f"Connected to Brain at {self._brain_address}")

    async def disconnect(self) -> None:
        """断开连接."""
        if not self._connected:
            return

        self._registered = False

        if self._socket:
            self._socket.close()
            self._socket = None

        if self._context:
            self._context.term()
            self._context = None

        self._connected = False
        logger.info("Disconnected from Brain")

    async def reconnect(self) -> bool:
        """重新连接到Brain.

        Returns:
            是否成功重连
        """
        if self._reconnecting:
            return False

        self._reconnecting = True
        logger.info(f"Attempting to reconnect to Brain (attempt {self._reconnect_count + 1}/{self._max_reconnect_attempts})")

        try:
            # 先断开
            await self.disconnect()

            # 尝试重连
            for attempt in range(self._max_reconnect_attempts):
                self._reconnect_count = attempt + 1

                try:
                    await self.connect()

                    # 重新注册
                    if self._registered:
                        success = await self.register(
                            capabilities=self._capabilities,
                            max_concurrent_tasks=self._max_concurrent_tasks,
                        )
                        if not success:
                            logger.warning("Re-registration failed, will retry...")
                            await asyncio.sleep(self._reconnect_interval)
                            continue

                    logger.info("Successfully reconnected to Brain")
                    return True

                except Exception as e:
                    logger.warning(f"Reconnect attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self._reconnect_interval * (attempt + 1))

            logger.error("Max reconnect attempts reached, giving up")
            return False

        finally:
            self._reconnecting = False

    async def _ensure_connected(self) -> None:
        """确保已连接，如果断开则尝试重连."""
        if not self._connected:
            success = await self.reconnect()
            if not success:
                raise ConnectionError("Cannot connect to Brain server")

    async def send(self, message: BrainMessage, retry_on_failure: bool = True) -> None:
        """发送消息.

        Args:
            message: 消息对象
            retry_on_failure: 失败时是否重试
        """
        await self._ensure_connected()

        if not self._socket:
            raise RuntimeError("Not connected to Brain")

        try:
            data = message.model_dump_json().encode()
            await self._socket.send(data)
        except zmq.ZMQError as e:
            logger.error(f"Send failed: {e}")
            if retry_on_failure:
                # 尝试重连并重发
                if await self.reconnect():
                    await self.send(message, retry_on_failure=False)
                else:
                    raise
            else:
                raise

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
        import platform as plat
        import socket

        # 保存注册信息（用于重连后重新注册）
        self._capabilities = capabilities or []
        self._max_concurrent_tasks = max_concurrent_tasks

        payload = WorkerRegisterMessage.Payload(
            worker_id=self._worker_id,
            hostname=socket.gethostname(),
            platform=plat.system().lower(),
            capabilities=self._capabilities,
            max_concurrent_tasks=max_concurrent_tasks,
        )
        message = WorkerRegisterMessage(
            msg_id=self._generate_request_id(),
            sender=self._worker_id,
            payload=payload,
        )

        await self.send(message, retry_on_failure=False)

        # 等待确认
        response = await self.receive(timeout=10)
        if response and response.get("type") == "register_ack":
            self._registered = True
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
        payload = HeartbeatMessage.Payload(
            worker_id=self._worker_id,
            status="idle",
            current_task_id=current_task_id,
        )
        message = HeartbeatMessage(
            msg_id=self._generate_request_id(),
            sender=self._worker_id,
            payload=payload,
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
        # 使用心跳消息传递状态（简化实现）
        payload = HeartbeatMessage.Payload(
            worker_id=self._worker_id,
            status=status.value,
            current_task_id=task_id,
            task_progress=progress or 0.0,
        )
        msg = HeartbeatMessage(
            msg_id=self._generate_request_id(),
            sender=self._worker_id,
            payload=payload,
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
        payload = TaskResultMessage.Payload(
            task_id=task_id,
            success=result.success,
            output=result.output or "",
            error=result.error,
            files_modified=result.files_modified,
            git_commits=result.git_commits,
            tokens_used=result.tokens_used,
            duration_seconds=result.duration_seconds,
            model_used=result.model_used,
        )
        msg = TaskResultMessage(
            msg_id=self._generate_request_id(),
            sender=self._worker_id,
            payload=payload,
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

        # 假设请求当前任务的上下文
        payload = ContextGetRequest.Payload(
            worker_id=self._worker_id,
            task_id=request_id,  # 使用request_id作为临时task_id
        )
        msg = ContextGetRequest(
            msg_id=request_id,
            sender=self._worker_id,
            payload=payload,
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
        payload = ContextUpdateRequest.Payload(
            worker_id=self._worker_id,
            task_id="",  # 全局上下文更新
            messages=[{"key": key, "value": value}],
        )
        msg = ContextUpdateRequest(
            msg_id=self._generate_request_id(),
            sender=self._worker_id,
            payload=payload,
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

        payload = SafetyCheckRequest.Payload(
            worker_id=self._worker_id,
            task_id=request_id,
            command_type=command_type,
            command=command,
            args=args or {},
        )
        msg = SafetyCheckRequest(
            msg_id=request_id,
            sender=self._worker_id,
            payload=payload,
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
