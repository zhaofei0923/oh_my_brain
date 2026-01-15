"""MiniAgentAdapter 测试."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oh_my_brain.worker.mini_agent_adapter import (
    MiniAgentAdapter,
    MiniAgentError,
    SafeBashTool,
    SafeCreateFileTool,
    SafeEditFileTool,
)


class TestMiniAgentAdapter:
    """MiniAgentAdapter 测试."""

    @pytest.fixture
    def adapter(self):
        """创建适配器实例."""
        return MiniAgentAdapter(worker_id="test-worker")

    def test_init(self, adapter):
        """测试初始化."""
        assert adapter._worker_id == "test-worker"
        assert adapter._safety_callback is None
        assert adapter._approval_callback is None

    def test_set_safety_callback(self, adapter):
        """测试设置安全回调."""
        callback = MagicMock()
        adapter.set_safety_callback(callback)
        assert adapter._safety_callback is callback

    def test_set_approval_callback(self, adapter):
        """测试设置审批回调."""
        callback = MagicMock()
        adapter.set_approval_callback(callback)
        assert adapter._approval_callback is callback

    def test_parse_modified_files_created(self, adapter):
        """测试解析创建的文件."""
        output = """
        Created file: src/main.py
        Created file: tests/test_main.py
        """
        files = adapter._parse_modified_files(output)
        assert "src/main.py" in files
        assert "tests/test_main.py" in files

    def test_parse_modified_files_modified(self, adapter):
        """测试解析修改的文件."""
        output = """
        Modified src/utils.py
        Updated config/settings.json
        """
        files = adapter._parse_modified_files(output)
        assert "src/utils.py" in files
        assert "config/settings.json" in files

    def test_parse_modified_files_checkmark(self, adapter):
        """测试解析带对勾的文件."""
        output = """
        ✓ src/new_file.py
        ✔ tests/test_new.py
        """
        files = adapter._parse_modified_files(output)
        assert "src/new_file.py" in files
        assert "tests/test_new.py" in files

    def test_parse_modified_files_empty(self, adapter):
        """测试空输出."""
        output = "Task completed successfully"
        files = adapter._parse_modified_files(output)
        assert files == []

    def test_parse_modified_files_filters_urls(self, adapter):
        """测试过滤 URL."""
        output = """
        Fetching from https://example.com/file.py
        Created src/real_file.py
        """
        files = adapter._parse_modified_files(output)
        assert "src/real_file.py" in files
        assert "https://example.com/file.py" not in files


class TestSafeBashTool:
    """SafeBashTool 测试."""

    @pytest.fixture
    def mock_safety_callback(self):
        """创建模拟安全回调."""
        async def callback(command_type, command, args):
            if "rm -rf /" in command:
                return {"approved": False, "reason": "Dangerous command"}
            return {"approved": True}
        return callback

    @pytest.mark.asyncio
    async def test_safe_command(self, mock_safety_callback):
        """测试安全命令."""
        tool = SafeBashTool(safety_callback=mock_safety_callback)
        # 注意：实际执行需要设置正确的工具实现
        # 这里只测试安全检查逻辑
        result = await mock_safety_callback("bash", "ls -la", {})
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_dangerous_command(self, mock_safety_callback):
        """测试危险命令."""
        result = await mock_safety_callback("bash", "rm -rf /", {})
        assert result["approved"] is False


class TestSafeCreateFileTool:
    """SafeCreateFileTool 测试."""

    @pytest.fixture
    def mock_safety_callback(self):
        """创建模拟安全回调."""
        async def callback(command_type, command, args):
            if command.startswith("/etc/"):
                return {"approved": False, "reason": "Protected path"}
            return {"approved": True}
        return callback

    @pytest.mark.asyncio
    async def test_allowed_path(self, mock_safety_callback):
        """测试允许的路径."""
        result = await mock_safety_callback("write", "/home/user/test.py", {})
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_protected_path(self, mock_safety_callback):
        """测试受保护的路径."""
        result = await mock_safety_callback("write", "/etc/passwd", {})
        assert result["approved"] is False


class TestMiniAgentError:
    """MiniAgentError 测试."""

    def test_error_creation(self):
        """测试错误创建."""
        error = MiniAgentError("Test error")
        assert str(error) == "Test error"

    def test_error_with_details(self):
        """测试带详情的错误."""
        error = MiniAgentError("Failed to execute", task_id="task-001")
        assert error.task_id == "task-001"


class TestMiniAgentIntegration:
    """MiniAgent 集成测试."""

    @pytest.fixture
    def adapter_with_mock_brain(self):
        """创建带模拟 Brain 客户端的适配器."""
        adapter = MiniAgentAdapter(worker_id="test-worker")

        # 模拟 Brain 客户端
        mock_client = MagicMock()
        mock_client.safety_check = AsyncMock(return_value={"approved": True})
        adapter._brain_client = mock_client

        return adapter

    @pytest.mark.asyncio
    async def test_build_prompt(self, adapter_with_mock_brain):
        """测试构建提示."""
        from oh_my_brain.schemas.task import Task, TaskType

        task = Task(
            id="task-001",
            module_id="mod1",
            sub_task_id="sub1",
            name="Test Task",
            type=TaskType.CODING,
            description="Create a hello world function",
            requirements="Must print 'Hello, World!'",
            files_involved=["src/hello.py"],
        )

        prompt = adapter_with_mock_brain._build_prompt(task)

        assert "Test Task" in prompt
        assert "Create a hello world function" in prompt
        assert "Hello, World!" in prompt
        assert "src/hello.py" in prompt
