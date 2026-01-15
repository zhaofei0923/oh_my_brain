"""Brain主服务.

Brain是整个系统的中央协调器，负责：
1. 接收Worker注册和心跳
2. 分配任务给Worker
3. 管理上下文和状态
4. 协调Git操作
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from oh_my_brain.brain.context_manager import ContextManager
from oh_my_brain.brain.git_manager import GitManager
from oh_my_brain.brain.model_router import ModelRouter
from oh_my_brain.brain.safety_checker import SafetyChecker
from oh_my_brain.brain.task_scheduler import TaskScheduler
from oh_my_brain.protocol.messages import (
    BrainMessage,
    ContextGetRequest,
    ContextGetResponse,
    ContextUpdateRequest,
    HeartbeatMessage,
    MessageType,
    SafetyCheckRequest,
    SafetyCheckResponse,
    TaskAssignMessage,
    TaskResultMessage,
    WorkerRegisterMessage,
)
from oh_my_brain.protocol.transport import Transport, get_default_endpoint, get_transport
from oh_my_brain.schemas.config import BrainConfig

logger = logging.getLogger(__name__)


class WorkerInfo:
    """Worker信息."""

    def __init__(
        self,
        worker_id: str,
        identity: bytes,
        hostname: str,
        platform: str,
        capabilities: list[str],
    ):
        self.worker_id = worker_id
        self.identity = identity
        self.hostname = hostname
        self.platform = platform
        self.capabilities = capabilities
        self.status = "idle"  # idle, busy, paused
        self.current_task_id: str | None = None
        self.last_heartbeat = datetime.now()
        self.registered_at = datetime.now()

    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """检查Worker是否健康（心跳未超时）."""
        delta = (datetime.now() - self.last_heartbeat).total_seconds()
        return delta < timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "platform": self.platform,
            "status": self.status,
            "current_task_id": self.current_task_id,
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }


class BrainServer:
    """Brain主服务."""

    def __init__(self, config: BrainConfig | None = None):
        self.config = config or BrainConfig()
        self._transport: Transport | None = None
        self._running = False

        # Worker管理
        self._workers: dict[str, WorkerInfo] = {}  # worker_id -> WorkerInfo
        self._identity_map: dict[bytes, str] = {}  # identity -> worker_id

        # 子模块
        self._context_manager = ContextManager(
            redis_url=self.config.redis_url,
            prefix=self.config.redis_prefix,
            token_limit=self.config.context_token_limit,
        )
        self._task_scheduler = TaskScheduler()
        self._model_router: ModelRouter | None = None
        self._safety_checker = SafetyChecker(
            enabled=self.config.safety_check_enabled,
            dangerous_commands=self.config.dangerous_commands,
        )
        self._git_manager = GitManager(
            base_branch=self.config.git_base_branch,
            branch_prefix=self.config.git_branch_prefix,
        )

        # 消息处理器映射
        self._handlers: dict[MessageType, Any] = {
            MessageType.WORKER_REGISTER: self._handle_worker_register,
            MessageType.WORKER_UNREGISTER: self._handle_worker_unregister,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.TASK_RESULT: self._handle_task_result,
            MessageType.CONTEXT_GET: self._handle_context_get,
            MessageType.CONTEXT_UPDATE: self._handle_context_update,
            MessageType.SAFETY_CHECK: self._handle_safety_check,
        }

    async def start(self) -> None:
        """启动Brain服务."""
        logger.info("Starting Brain server...")

        # 初始化传输层
        self._transport = get_transport("brain")
        endpoint = get_default_endpoint()

        if self.config.transport == "tcp":
            endpoint = f"tcp://{self.config.host}:{self.config.port}"
        elif self.config.transport == "ipc":
            endpoint = "ipc:///tmp/oh-my-brain.sock"

        await self._transport.bind(endpoint)
        logger.info(f"Brain listening on {endpoint}")

        # 初始化子模块
        await self._context_manager.connect()

        # 加载模型配置
        if self.config.models_config_path.exists():
            self._model_router = ModelRouter.from_yaml(self.config.models_config_path)

        self._running = True

        # 启动心跳检查任务
        asyncio.create_task(self._heartbeat_check_loop())

        # 主消息循环
        await self._message_loop()

    async def stop(self) -> None:
        """停止Brain服务."""
        logger.info("Stopping Brain server...")
        self._running = False

        # 通知所有Worker关闭
        for worker_id, worker in self._workers.items():
            try:
                await self._send_shutdown(worker.identity)
            except Exception as e:
                logger.warning(f"Failed to send shutdown to {worker_id}: {e}")

        # 关闭连接
        await self._context_manager.disconnect()
        if self._transport:
            await self._transport.close()

        logger.info("Brain server stopped")

    async def _message_loop(self) -> None:
        """消息处理主循环."""
        while self._running:
            try:
                if self._transport is None:
                    break

                identity, data = await self._transport.recv()
                if identity is None:
                    continue

                try:
                    message = BrainMessage.from_bytes(data)
                    await self._handle_message(identity, message)
                except Exception as e:
                    logger.error(f"Failed to handle message: {e}")
                    await self._send_error(identity, str(e))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message loop: {e}")

    async def _handle_message(self, identity: bytes, message: BrainMessage) -> None:
        """处理收到的消息."""
        handler = self._handlers.get(message.msg_type)
        if handler:
            await handler(identity, message)
        else:
            logger.warning(f"Unknown message type: {message.msg_type}")

    async def _handle_worker_register(
        self, identity: bytes, message: WorkerRegisterMessage
    ) -> None:
        """处理Worker注册."""
        payload = message.payload
        worker_id = payload.worker_id

        worker = WorkerInfo(
            worker_id=worker_id,
            identity=identity,
            hostname=payload.hostname,
            platform=payload.platform,
            capabilities=payload.capabilities,
        )

        self._workers[worker_id] = worker
        self._identity_map[identity] = worker_id

        logger.info(f"Worker registered: {worker_id} from {payload.hostname}")

        # 发送确认
        await self._send_ack(identity, message.msg_id)

    async def _handle_worker_unregister(self, identity: bytes, message: BrainMessage) -> None:
        """处理Worker注销."""
        worker_id = self._identity_map.get(identity)
        if worker_id:
            del self._workers[worker_id]
            del self._identity_map[identity]
            logger.info(f"Worker unregistered: {worker_id}")

    async def _handle_heartbeat(self, identity: bytes, message: HeartbeatMessage) -> None:
        """处理心跳."""
        worker_id = message.payload.worker_id
        worker = self._workers.get(worker_id)

        if worker:
            worker.last_heartbeat = datetime.now()
            worker.status = message.payload.status
            worker.current_task_id = message.payload.current_task_id

    async def _handle_task_request(self, identity: bytes, message: BrainMessage) -> None:
        """处理任务请求."""
        worker_id = self._identity_map.get(identity)
        if not worker_id:
            return

        worker = self._workers.get(worker_id)
        if not worker:
            return

        # 从调度器获取下一个任务
        task = self._task_scheduler.get_next_task(worker_id)
        if task is None:
            # 没有任务，发送空响应
            await self._send_ack(identity, message.msg_id, {"task": None})
            return

        # 获取模型配置
        model_name = task.assigned_model or "default"
        model_config: dict[str, Any] = {}
        if self._model_router:
            model = self._model_router.get_model(model_name)
            if model:
                model_config = model.model_dump()

        # 创建Git分支
        branch_name = f"{self.config.git_branch_prefix}{task.id}"
        task.git_branch = branch_name

        # 获取初始上下文
        context = await self._context_manager.get_context(task.id)

        # 发送任务
        assign_msg = TaskAssignMessage(
            msg_id=str(uuid.uuid4()),
            sender="brain",
            payload=TaskAssignMessage.Payload(
                task_id=task.id,
                module_id=task.module_id,
                name=task.name,
                type=task.type.value,
                description=task.description,
                requirements=task.requirements,
                files_involved=task.files_involved,
                git_branch=branch_name,
                model_name=model_name,
                model_config=model_config,
                context=context,
            ),
        )

        await self._transport.send(assign_msg.to_bytes(), identity)  # type: ignore

        # 更新Worker状态
        worker.status = "busy"
        worker.current_task_id = task.id

    async def _handle_task_result(self, identity: bytes, message: TaskResultMessage) -> None:
        """处理任务结果."""
        payload = message.payload
        worker_id = self._identity_map.get(identity)

        if worker_id:
            worker = self._workers.get(worker_id)
            if worker:
                worker.status = "idle"
                worker.current_task_id = None

        # 更新任务状态
        if payload.success:
            self._task_scheduler.mark_completed(payload.task_id)
        else:
            self._task_scheduler.mark_failed(payload.task_id, payload.error or "Unknown error")

        # 更新上下文
        if payload.context_update:
            await self._context_manager.update_context(payload.task_id, payload.context_update)

        logger.info(
            f"Task {payload.task_id} completed: success={payload.success}, "
            f"tokens={payload.tokens_used}"
        )

    async def _handle_context_get(self, identity: bytes, message: ContextGetRequest) -> None:
        """处理上下文获取请求."""
        payload = message.payload
        context = await self._context_manager.get_context(payload.task_id)

        response = ContextGetResponse(
            msg_id=str(uuid.uuid4()),
            sender="brain",
            payload=ContextGetResponse.Payload(
                task_id=payload.task_id,
                messages=context.get("messages", []),
                notes=context.get("notes", []),
                system_prompt=context.get("system_prompt", ""),
                token_budget=self.config.context_token_limit,
            ),
        )

        await self._transport.send(response.to_bytes(), identity)  # type: ignore

    async def _handle_context_update(self, identity: bytes, message: ContextUpdateRequest) -> None:
        """处理上下文更新请求."""
        payload = message.payload
        await self._context_manager.update_context(
            payload.task_id,
            {
                "messages": payload.messages,
                "notes": payload.notes,
            },
        )
        await self._send_ack(identity, message.msg_id)

    async def _handle_safety_check(self, identity: bytes, message: SafetyCheckRequest) -> None:
        """处理安全检查请求."""
        payload = message.payload
        result = self._safety_checker.check(
            command_type=payload.command_type,
            command=payload.command,
            args=payload.args,
        )

        response = SafetyCheckResponse(
            msg_id=str(uuid.uuid4()),
            sender="brain",
            payload=SafetyCheckResponse.Payload(
                approved=result.approved,
                reason=result.reason,
                modified_command=result.modified_command,
            ),
        )

        await self._transport.send(response.to_bytes(), identity)  # type: ignore

    async def _send_ack(self, identity: bytes, msg_id: str, data: dict | None = None) -> None:
        """发送确认消息."""
        ack = BrainMessage(
            msg_type=MessageType.ACK,
            msg_id=str(uuid.uuid4()),
            sender="brain",
            payload={"ref_msg_id": msg_id, **(data or {})},
        )
        await self._transport.send(ack.to_bytes(), identity)  # type: ignore

    async def _send_error(self, identity: bytes, error: str) -> None:
        """发送错误消息."""
        msg = BrainMessage(
            msg_type=MessageType.ERROR,
            msg_id=str(uuid.uuid4()),
            sender="brain",
            payload={"error": error},
        )
        await self._transport.send(msg.to_bytes(), identity)  # type: ignore

    async def _send_shutdown(self, identity: bytes) -> None:
        """发送关闭命令."""
        msg = BrainMessage(
            msg_type=MessageType.SHUTDOWN,
            msg_id=str(uuid.uuid4()),
            sender="brain",
            payload={},
        )
        await self._transport.send(msg.to_bytes(), identity)  # type: ignore

    async def _heartbeat_check_loop(self) -> None:
        """心跳检查循环."""
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval_seconds)

            datetime.now()
            timeout = self.config.heartbeat_timeout_seconds

            for worker_id, worker in list(self._workers.items()):
                if not worker.is_healthy(timeout):
                    logger.warning(f"Worker {worker_id} heartbeat timeout")
                    # 处理Worker超时（重新分配任务等）
                    if worker.current_task_id:
                        self._task_scheduler.requeue_task(worker.current_task_id)
                    # 移除Worker
                    del self._workers[worker_id]
                    if worker.identity in self._identity_map:
                        del self._identity_map[worker.identity]

    def get_status(self) -> dict[str, Any]:
        """获取Brain状态."""
        return {
            "running": self._running,
            "workers": {wid: w.to_dict() for wid, w in self._workers.items()},
            "tasks": self._task_scheduler.get_status(),
        }
