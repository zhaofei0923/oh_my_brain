"""Worker基类."""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

from oh_my_brain.schemas.config import WorkerConfig
from oh_my_brain.schemas.task import Task, TaskResult, TaskStatus
from oh_my_brain.worker.brain_client import BrainClient

logger = logging.getLogger(__name__)


class WorkerBase(ABC):
    """Worker基类.

    职责：
    1. 与Brain通信（支持自动重连）
    2. 接收任务分配
    3. 执行任务（通过子类实现）
    4. 上报任务结果
    5. 断点续传支持
    """

    def __init__(self, config: WorkerConfig):
        self._config = config
        self._worker_id = config.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._brain_client = BrainClient(
            brain_address=config.brain_address,
            worker_id=self._worker_id,
            max_reconnect_attempts=getattr(config, 'max_reconnect_attempts', 10),
            reconnect_interval=getattr(config, 'reconnect_interval', 2.0),
        )

        self._running = False
        self._current_task: Task | None = None
        self._capabilities: list[str] = []

        # 错误计数（用于背压控制）
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

    @property
    def worker_id(self) -> str:
        """获取Worker ID."""
        return self._worker_id

    @property
    def is_running(self) -> bool:
        """是否正在运行."""
        return self._running

    @property
    def current_task(self) -> Task | None:
        """当前任务."""
        return self._current_task

    def set_capabilities(self, capabilities: list[str]) -> None:
        """设置Worker能力.

        Args:
            capabilities: 能力列表，如 ["python", "javascript", "docker"]
        """
        self._capabilities = capabilities

    async def start(self) -> None:
        """启动Worker."""
        self._running = True
        logger.info(f"Worker {self._worker_id} starting...")

        # 连接Brain
        await self._brain_client.connect()

        # 注册Worker
        await self._brain_client.register(
            capabilities=self._capabilities,
            max_concurrent_tasks=self._config.max_concurrent_tasks,
        )

        # 启动心跳
        asyncio.create_task(self._heartbeat_loop())

        # 启动任务处理循环
        await self._task_loop()

    async def stop(self) -> None:
        """停止Worker."""
        self._running = False
        await self._brain_client.disconnect()
        logger.info(f"Worker {self._worker_id} stopped")

    async def _heartbeat_loop(self) -> None:
        """心跳循环."""
        while self._running:
            try:
                await self._brain_client.send_heartbeat(
                    current_task_id=self._current_task.id if self._current_task else None,
                )
                # 成功则重置错误计数
                self._consecutive_errors = 0
            except ConnectionError as e:
                logger.error(f"Heartbeat connection error: {e}")
                self._consecutive_errors += 1
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error("Too many consecutive errors, attempting reconnect...")
                    await self._brain_client.reconnect()
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                self._consecutive_errors += 1

            await asyncio.sleep(self._config.heartbeat_interval_seconds)

    async def _task_loop(self) -> None:
        """任务处理循环."""
        while self._running:
            try:
                # 检查连接状态
                if not self._brain_client.is_connected:
                    logger.warning("Lost connection to Brain, attempting to reconnect...")
                    success = await self._brain_client.reconnect()
                    if not success:
                        logger.error("Failed to reconnect, waiting before retry...")
                        await asyncio.sleep(5)
                        continue

                # 等待任务分配
                message = await self._brain_client.receive()

                if message is None:
                    await asyncio.sleep(0.1)
                    continue

                # 成功接收消息，重置错误计数
                self._consecutive_errors = 0

                if message.get("type") == "task_assign":
                    task_data = message.get("task")
                    if task_data:
                        task = Task.model_validate(task_data)
                        await self._handle_task(task)

                elif message.get("type") == "shutdown":
                    logger.info("Received shutdown signal")
                    await self.stop()
                    break

                elif message.get("type") == "ping":
                    # 响应 ping 请求
                    logger.debug("Received ping, responding...")

            except ConnectionError as e:
                logger.error(f"Connection error in task loop: {e}")
                self._consecutive_errors += 1
                if self._consecutive_errors >= self._max_consecutive_errors:
                    await self._brain_client.reconnect()
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task loop error: {e}")
                self._consecutive_errors += 1
                await asyncio.sleep(1)

    async def _handle_task(self, task: Task) -> None:
        """处理任务.

        Args:
            task: 任务对象
        """
        self._current_task = task
        logger.info(f"Handling task: {task.id}")

        try:
            # 通知Brain任务开始
            await self._brain_client.report_task_status(
                task_id=task.id,
                status=TaskStatus.RUNNING,
            )

            # 执行任务
            result = await self.execute_task(task)

            # 上报结果
            await self._brain_client.report_task_result(task.id, result)

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            await self._brain_client.report_task_result(
                task_id=task.id,
                result=TaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(e),
                ),
            )

        finally:
            self._current_task = None

    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """执行任务.

        子类必须实现此方法。

        Args:
            task: 任务对象

        Returns:
            任务结果
        """
        pass

    async def request_context(self, keys: list[str]) -> dict[str, Any]:
        """向Brain请求上下文.

        Args:
            keys: 需要的上下文键列表

        Returns:
            上下文数据
        """
        return await self._brain_client.request_context(keys)

    async def update_context(self, key: str, value: Any) -> None:
        """更新上下文.

        Args:
            key: 上下文键
            value: 上下文值
        """
        await self._brain_client.update_context(key, value)

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
        return await self._brain_client.request_safety_check(
            command_type=command_type,
            command=command,
            args=args,
        )

    async def save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """保存任务检查点.

        用于断点续传，子类可以在任务执行过程中定期调用此方法保存进度。

        Args:
            checkpoint: 检查点数据
        """
        if self._current_task:
            await self._brain_client.update_context(
                key=f"checkpoint:{self._current_task.id}",
                value=checkpoint,
            )
            logger.debug(f"Saved checkpoint for task {self._current_task.id}")

    async def load_checkpoint(self) -> dict[str, Any] | None:
        """加载任务检查点.

        Returns:
            检查点数据，如果没有则返回 None
        """
        if self._current_task:
            context = await self._brain_client.request_context(
                [f"checkpoint:{self._current_task.id}"]
            )
            return context.get(f"checkpoint:{self._current_task.id}")
        return None

    async def report_progress(self, progress: float, message: str | None = None) -> None:
        """上报任务进度.

        Args:
            progress: 进度 (0.0 - 1.0)
            message: 可选的状态消息
        """
        if self._current_task:
            await self._brain_client.report_task_status(
                task_id=self._current_task.id,
                status=TaskStatus.RUNNING,
                progress=progress,
                message=message,
            )
