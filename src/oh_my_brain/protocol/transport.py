"""传输层实现.

支持IPC（Linux/macOS）和TCP（Windows）两种传输方式。
"""

import platform
from abc import ABC, abstractmethod

import zmq
import zmq.asyncio


class Transport(ABC):
    """传输层抽象基类."""

    @abstractmethod
    async def bind(self, endpoint: str) -> None:
        """绑定到端点（服务端）."""
        ...

    @abstractmethod
    async def connect(self, endpoint: str) -> None:
        """连接到端点（客户端）."""
        ...

    @abstractmethod
    async def send(self, data: bytes, identity: bytes | None = None) -> None:
        """发送数据."""
        ...

    @abstractmethod
    async def recv(self) -> tuple[bytes | None, bytes]:
        """接收数据.

        Returns:
            (identity, data) - identity为发送者标识（ROUTER模式），data为数据
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """关闭连接."""
        ...


class ZmqRouterTransport(Transport):
    """ZeroMQ ROUTER传输层（Brain端使用）.

    支持多Worker连接，可识别每个Worker的身份。
    """

    def __init__(self) -> None:
        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None

    async def bind(self, endpoint: str) -> None:
        """绑定到端点."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)  # 发送给未知Worker时报错
        self._socket.bind(endpoint)

    async def connect(self, endpoint: str) -> None:
        """ROUTER不支持connect."""
        raise NotImplementedError("ROUTER transport should use bind()")

    async def send(self, data: bytes, identity: bytes | None = None) -> None:
        """发送数据到指定Worker."""
        if self._socket is None:
            raise RuntimeError("Transport not initialized")
        if identity is None:
            raise ValueError("ROUTER transport requires identity")
        await self._socket.send_multipart([identity, b"", data])

    async def recv(self) -> tuple[bytes | None, bytes]:
        """接收数据.

        Returns:
            (identity, data)
        """
        if self._socket is None:
            raise RuntimeError("Transport not initialized")
        parts = await self._socket.recv_multipart()
        # ROUTER格式: [identity, empty, data]
        if len(parts) >= 3:
            return parts[0], parts[2]
        elif len(parts) == 2:
            return parts[0], parts[1]
        else:
            return None, parts[0]

    async def close(self) -> None:
        """关闭连接."""
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()


class ZmqDealerTransport(Transport):
    """ZeroMQ DEALER传输层（Worker端使用）.

    连接到Brain ROUTER。
    """

    def __init__(self, identity: str) -> None:
        self._identity = identity.encode("utf-8")
        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None

    async def bind(self, endpoint: str) -> None:
        """DEALER不支持bind."""
        raise NotImplementedError("DEALER transport should use connect()")

    async def connect(self, endpoint: str) -> None:
        """连接到Brain."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, self._identity)
        self._socket.connect(endpoint)

    async def send(self, data: bytes, identity: bytes | None = None) -> None:
        """发送数据到Brain."""
        if self._socket is None:
            raise RuntimeError("Transport not initialized")
        await self._socket.send_multipart([b"", data])

    async def recv(self) -> tuple[bytes | None, bytes]:
        """接收数据.

        Returns:
            (None, data) - DEALER模式不需要identity
        """
        if self._socket is None:
            raise RuntimeError("Transport not initialized")
        parts = await self._socket.recv_multipart()
        # DEALER格式: [empty, data]
        if len(parts) >= 2:
            return None, parts[1]
        else:
            return None, parts[0]

    async def close(self) -> None:
        """关闭连接."""
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()


def get_default_endpoint(transport_type: str = "auto") -> str:
    """获取默认端点.

    Args:
        transport_type: 传输类型 (auto, ipc, tcp)

    Returns:
        端点地址
    """
    if transport_type == "tcp":
        return "tcp://127.0.0.1:15555"
    elif transport_type == "ipc":
        return "ipc:///tmp/oh-my-brain.sock"
    else:  # auto
        if platform.system() == "Windows":
            return "tcp://127.0.0.1:15555"
        else:
            return "ipc:///tmp/oh-my-brain.sock"


def get_transport(
    role: str,
    identity: str = "",
    transport_type: str = "auto",
) -> Transport:
    """获取传输层实例.

    Args:
        role: 角色 (brain, worker)
        identity: Worker标识（仅Worker需要）
        transport_type: 传输类型 (auto, ipc, tcp)

    Returns:
        传输层实例
    """
    if role == "brain":
        return ZmqRouterTransport()
    elif role == "worker":
        if not identity:
            raise ValueError("Worker requires identity")
        return ZmqDealerTransport(identity)
    else:
        raise ValueError(f"Unknown role: {role}")


def convert_endpoint(endpoint: str, transport_type: str = "auto") -> str:
    """转换端点地址以适配平台.

    Args:
        endpoint: 原始端点
        transport_type: 传输类型

    Returns:
        转换后的端点
    """
    if transport_type == "tcp":
        # 强制使用TCP
        if endpoint.startswith("ipc://"):
            return "tcp://127.0.0.1:15555"
        return endpoint
    elif transport_type == "ipc":
        # 强制使用IPC（仅Linux/macOS）
        if platform.system() == "Windows":
            raise ValueError("IPC transport not supported on Windows")
        if endpoint.startswith("tcp://"):
            return "ipc:///tmp/oh-my-brain.sock"
        return endpoint
    else:  # auto
        if platform.system() == "Windows" and endpoint.startswith("ipc://"):
            # Windows不支持IPC，转换为TCP
            return "tcp://127.0.0.1:15555"
        return endpoint
