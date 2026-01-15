"""HTTP 服务器用于暴露 Prometheus 指标和健康检查端点.

这是一个轻量级的 HTTP 服务器，不依赖外部框架。
"""

import asyncio
import logging
from http import HTTPStatus
from typing import Any

from oh_my_brain.brain.metrics import get_metrics_endpoint

logger = logging.getLogger(__name__)


class MetricsServer:
    """Prometheus 指标 HTTP 服务器."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self.host = host
        self.port = port
        self._server: asyncio.Server | None = None
        self._running = False
        self._status_provider: Any = None  # BrainServer 实例

    def set_status_provider(self, provider: Any) -> None:
        """设置状态提供者（通常是 BrainServer）."""
        self._status_provider = provider

    async def start(self) -> None:
        """启动 HTTP 服务器."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )
        logger.info(f"Metrics server listening on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """停止 HTTP 服务器."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Metrics server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """处理 HTTP 连接."""
        try:
            # 读取请求行
            request_line = await reader.readline()
            if not request_line:
                return

            request_str = request_line.decode("utf-8", errors="ignore")
            parts = request_str.strip().split(" ")
            if len(parts) < 2:
                await self._send_response(writer, HTTPStatus.BAD_REQUEST, "Bad Request")
                return

            method, path = parts[0], parts[1]

            # 读取并忽略请求头
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n" or not line:
                    break

            # 路由处理
            if method == "GET":
                if path == "/metrics":
                    await self._handle_metrics(writer)
                elif path == "/health" or path == "/healthz":
                    await self._handle_health(writer)
                elif path == "/ready" or path == "/readyz":
                    await self._handle_ready(writer)
                elif path == "/status":
                    await self._handle_status(writer)
                else:
                    await self._send_response(writer, HTTPStatus.NOT_FOUND, "Not Found")
            else:
                await self._send_response(
                    writer, HTTPStatus.METHOD_NOT_ALLOWED, "Method Not Allowed"
                )

        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            try:
                await self._send_response(
                    writer, HTTPStatus.INTERNAL_SERVER_ERROR, str(e)
                )
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_metrics(self, writer: asyncio.StreamWriter) -> None:
        """处理 /metrics 请求."""
        content = get_metrics_endpoint()
        await self._send_response(
            writer,
            HTTPStatus.OK,
            content,
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """处理 /health 请求."""
        import json

        health = {"status": "healthy"}
        await self._send_response(
            writer,
            HTTPStatus.OK,
            json.dumps(health),
            content_type="application/json",
        )

    async def _handle_ready(self, writer: asyncio.StreamWriter) -> None:
        """处理 /ready 请求."""
        import json

        ready = True
        if self._status_provider:
            status = self._status_provider.get_status()
            ready = status.get("running", False)

        if ready:
            await self._send_response(
                writer,
                HTTPStatus.OK,
                json.dumps({"ready": True}),
                content_type="application/json",
            )
        else:
            await self._send_response(
                writer,
                HTTPStatus.SERVICE_UNAVAILABLE,
                json.dumps({"ready": False}),
                content_type="application/json",
            )

    async def _handle_status(self, writer: asyncio.StreamWriter) -> None:
        """处理 /status 请求."""
        import json

        if self._status_provider:
            status = self._status_provider.get_status()
            # 处理不可序列化的对象
            safe_status = self._make_json_safe(status)
            await self._send_response(
                writer,
                HTTPStatus.OK,
                json.dumps(safe_status, indent=2),
                content_type="application/json",
            )
        else:
            await self._send_response(
                writer,
                HTTPStatus.OK,
                json.dumps({"status": "no provider"}),
                content_type="application/json",
            )

    def _make_json_safe(self, obj: Any) -> Any:
        """将对象转换为可 JSON 序列化的格式."""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, "isoformat"):  # datetime
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return self._make_json_safe(obj.__dict__)
        else:
            return str(obj)

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status: HTTPStatus,
        body: str,
        content_type: str = "text/plain",
    ) -> None:
        """发送 HTTP 响应."""
        body_bytes = body.encode("utf-8")
        response = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(response.encode("utf-8"))
        writer.write(body_bytes)
        await writer.drain()


async def run_metrics_server(
    host: str = "0.0.0.0",
    port: int = 9090,
    status_provider: Any = None,
) -> MetricsServer:
    """启动指标服务器."""
    server = MetricsServer(host, port)
    if status_provider:
        server.set_status_provider(status_provider)
    await server.start()
    return server
