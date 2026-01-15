"""平台适配器抽象基类."""

import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PlatformAdapter(ABC):
    """平台适配器抽象基类.

    封装不同操作系统的差异，提供统一接口。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """平台名称."""
        ...

    @property
    @abstractmethod
    def default_transport(self) -> str:
        """默认传输协议 (ipc 或 tcp)."""
        ...

    @property
    @abstractmethod
    def default_endpoint(self) -> str:
        """默认通信端点."""
        ...

    @abstractmethod
    def get_temp_dir(self) -> Path:
        """获取临时目录."""
        ...

    @abstractmethod
    def get_config_dir(self) -> Path:
        """获取配置目录."""
        ...

    @abstractmethod
    def get_data_dir(self) -> Path:
        """获取数据目录."""
        ...

    @abstractmethod
    def setup_multiprocessing(self) -> None:
        """配置多进程启动方式."""
        ...

    @abstractmethod
    def create_process(
        self,
        target: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> mp.Process:
        """创建进程."""
        ...

    @abstractmethod
    def terminate_process(self, process: mp.Process, timeout: float = 5.0) -> bool:
        """优雅终止进程.

        Args:
            process: 要终止的进程
            timeout: 等待超时时间

        Returns:
            是否成功优雅终止
        """
        ...

    @abstractmethod
    def get_process_info(self, process: mp.Process) -> dict[str, Any]:
        """获取进程信息.

        Returns:
            包含 pid, memory_mb, cpu_percent 等信息的字典
        """
        ...

    def convert_path(self, path: str | Path) -> Path:
        """转换路径为平台格式.

        默认实现，子类可覆盖。
        """
        return Path(path)

    def is_path_safe(self, path: str | Path) -> bool:
        """检查路径是否安全（不在系统目录内）.

        默认实现，子类可覆盖。
        """
        danger_paths = ["/", "/etc", "/usr", "/bin", "/sbin", "/var"]
        path_str = str(Path(path).resolve())
        return not any(path_str == dp or path_str.startswith(dp + "/") for dp in danger_paths)
