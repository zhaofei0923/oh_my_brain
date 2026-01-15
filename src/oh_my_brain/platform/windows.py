"""Windows平台适配器."""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from oh_my_brain.platform.base import PlatformAdapter


class WindowsAdapter(PlatformAdapter):
    """Windows平台适配器."""

    @property
    def name(self) -> str:
        return "windows"

    @property
    def default_transport(self) -> str:
        # Windows不支持IPC，使用TCP
        return "tcp"

    @property
    def default_endpoint(self) -> str:
        return "tcp://127.0.0.1:15555"

    def get_temp_dir(self) -> Path:
        temp = os.environ.get("TEMP", os.environ.get("TMP", ""))
        if temp:
            return Path(temp) / "oh-my-brain"
        return Path.home() / "AppData" / "Local" / "Temp" / "oh-my-brain"

    def get_config_dir(self) -> Path:
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "oh-my-brain"
        return Path.home() / "AppData" / "Roaming" / "oh-my-brain"

    def get_data_dir(self) -> Path:
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            return Path(localappdata) / "oh-my-brain"
        return Path.home() / "AppData" / "Local" / "oh-my-brain"

    def setup_multiprocessing(self) -> None:
        """Windows只能使用spawn."""
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # 已经设置过

    def create_process(
        self,
        target: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> mp.Process:
        kwargs = kwargs or {}
        return mp.Process(target=target, args=args, kwargs=kwargs)

    def terminate_process(self, process: mp.Process, timeout: float = 5.0) -> bool:
        """终止进程.

        Windows不支持SIGTERM，直接使用terminate()。
        """
        if not process.is_alive():
            return True

        try:
            process.terminate()
            process.join(timeout=timeout)

            if process.is_alive():
                # 超时，尝试kill
                process.kill()
                process.join(timeout=1.0)
                return False

            return True
        except Exception:
            return False

    def get_process_info(self, process: mp.Process) -> dict[str, Any]:
        """获取进程信息."""
        info: dict[str, Any] = {
            "pid": process.pid,
            "alive": process.is_alive(),
            "exitcode": process.exitcode,
        }

        # Windows下获取内存信息需要额外库（如psutil）
        # 这里只返回基础信息，后续可扩展
        return info

    def convert_path(self, path: str | Path) -> Path:
        """转换路径，处理Windows路径分隔符."""
        path_str = str(path)
        # 统一使用正斜杠
        path_str = path_str.replace("\\", "/")
        return Path(path_str)

    def is_path_safe(self, path: str | Path) -> bool:
        """检查路径是否安全."""
        path_obj = Path(path).resolve()
        path_str = str(path_obj).lower()

        # Windows危险路径
        danger_patterns = [
            "c:\\windows",
            "c:\\program files",
            "c:\\program files (x86)",
            "c:\\programdata",
            "c:\\users\\public",
        ]

        for pattern in danger_patterns:
            if path_str.startswith(pattern.lower()):
                return False

        # 检查是否是驱动器根目录
        if len(path_str) <= 3 and path_str.endswith(":\\"):
            return False

        return True
