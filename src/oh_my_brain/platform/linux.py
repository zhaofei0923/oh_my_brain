"""Linux平台适配器."""

import multiprocessing as mp
import os
import signal
from pathlib import Path
from typing import Any

from oh_my_brain.platform.base import PlatformAdapter


class LinuxAdapter(PlatformAdapter):
    """Linux平台适配器."""

    @property
    def name(self) -> str:
        return "linux"

    @property
    def default_transport(self) -> str:
        return "ipc"

    @property
    def default_endpoint(self) -> str:
        return "ipc:///tmp/oh-my-brain.sock"

    def get_temp_dir(self) -> Path:
        return Path("/tmp/oh-my-brain")

    def get_config_dir(self) -> Path:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
        if xdg_config:
            return Path(xdg_config) / "oh-my-brain"
        return Path.home() / ".config" / "oh-my-brain"

    def get_data_dir(self) -> Path:
        xdg_data = os.environ.get("XDG_DATA_HOME", "")
        if xdg_data:
            return Path(xdg_data) / "oh-my-brain"
        return Path.home() / ".local" / "share" / "oh-my-brain"

    def setup_multiprocessing(self) -> None:
        """Linux可以使用fork，但为了一致性使用spawn."""
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
        """优雅终止进程，先发SIGTERM，超时后SIGKILL."""
        if not process.is_alive():
            return True

        pid = process.pid
        if pid is None:
            return False

        try:
            # 先尝试优雅终止
            os.kill(pid, signal.SIGTERM)
            process.join(timeout=timeout)

            if process.is_alive():
                # 超时，强制终止
                os.kill(pid, signal.SIGKILL)
                process.join(timeout=1.0)
                return False

            return True
        except (ProcessLookupError, OSError):
            return True  # 进程已不存在

    def get_process_info(self, process: mp.Process) -> dict[str, Any]:
        """获取进程信息."""
        info: dict[str, Any] = {
            "pid": process.pid,
            "alive": process.is_alive(),
            "exitcode": process.exitcode,
        }

        if process.pid and process.is_alive():
            try:
                # 尝试读取 /proc 信息
                with open(f"/proc/{process.pid}/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            # 内存使用（KB）
                            parts = line.split()
                            if len(parts) >= 2:
                                info["memory_mb"] = int(parts[1]) / 1024
                            break
            except (FileNotFoundError, PermissionError):
                pass

        return info

    def is_path_safe(self, path: str | Path) -> bool:
        """检查路径是否安全."""
        danger_paths = [
            "/",
            "/etc",
            "/usr",
            "/bin",
            "/sbin",
            "/var",
            "/boot",
            "/lib",
            "/lib64",
            "/root",
            "/sys",
            "/proc",
            "/dev",
        ]
        path_str = str(Path(path).resolve())

        # 检查是否是危险路径或其子路径
        for dp in danger_paths:
            if path_str == dp:
                return False
            # 允许 /tmp 和 /home 下的路径
            if dp not in ["/tmp", "/home"] and path_str.startswith(dp + "/"):
                return False

        return True
