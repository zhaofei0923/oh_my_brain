"""平台适配器测试."""

import sys

import pytest

from oh_my_brain.platform.base import PlatformAdapter
from oh_my_brain.platform.detect import get_platform, get_platform_adapter


class TestPlatformDetection:
    """测试平台检测."""

    def test_get_platform(self):
        """测试获取平台."""
        platform = get_platform()
        assert platform in ["linux", "darwin", "windows"]

    def test_get_platform_adapter(self):
        """测试获取平台适配器."""
        adapter = get_platform_adapter()
        assert isinstance(adapter, PlatformAdapter)

    def test_platform_adapter_methods(self):
        """测试平台适配器方法."""
        adapter = get_platform_adapter()

        # 测试获取默认传输
        transport = adapter.default_transport
        assert transport in ["ipc", "tcp"]

        # 测试基本属性
        assert hasattr(adapter, "setup_multiprocessing")
        assert hasattr(adapter, "create_process")

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_linux_specific(self):
        """测试Linux特定功能."""
        from oh_my_brain.platform.linux import LinuxAdapter

        adapter = LinuxAdapter()
        assert adapter.default_transport == "ipc"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_windows_specific(self):
        """测试Windows特定功能."""
        from oh_my_brain.platform.windows import WindowsAdapter

        adapter = WindowsAdapter()
        assert adapter.default_transport == "tcp"
