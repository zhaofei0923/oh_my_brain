"""平台检测与适配器获取."""

import platform
from functools import lru_cache

from oh_my_brain.platform.base import PlatformAdapter


@lru_cache(maxsize=1)
def get_platform() -> str:
    """获取当前平台名称.

    Returns:
        平台名称: linux, windows, darwin
    """
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    elif system == "darwin":
        return "darwin"
    else:
        # 未知平台，尝试作为Linux处理
        return "linux"


def is_wsl() -> bool:
    """检测是否在WSL环境中运行."""
    try:
        with open("/proc/version") as f:
            version = f.read().lower()
            return "microsoft" in version or "wsl" in version
    except FileNotFoundError:
        return False


@lru_cache(maxsize=1)
def get_platform_adapter() -> PlatformAdapter:
    """获取当前平台的适配器.

    Returns:
        平台适配器实例
    """
    plat = get_platform()

    if plat == "linux" or plat == "darwin":
        from oh_my_brain.platform.linux import LinuxAdapter

        return LinuxAdapter()
    elif plat == "windows":
        from oh_my_brain.platform.windows import WindowsAdapter

        return WindowsAdapter()
    else:
        # 默认使用Linux适配器
        from oh_my_brain.platform.linux import LinuxAdapter

        return LinuxAdapter()


def get_default_transport() -> str:
    """获取当前平台的默认传输协议."""
    return get_platform_adapter().default_transport


def get_default_endpoint() -> str:
    """获取当前平台的默认通信端点."""
    return get_platform_adapter().default_endpoint
