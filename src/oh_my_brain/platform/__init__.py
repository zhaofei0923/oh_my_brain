"""平台适配层."""

from oh_my_brain.platform.base import PlatformAdapter
from oh_my_brain.platform.detect import get_platform, get_platform_adapter

__all__ = [
    "PlatformAdapter",
    "get_platform",
    "get_platform_adapter",
]
