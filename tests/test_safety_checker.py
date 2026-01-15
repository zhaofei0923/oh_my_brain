"""安全检查器和审批流程测试."""

import asyncio
import pytest

from oh_my_brain.brain.safety_checker import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalStatus,
    SafetyChecker,
    SafetyCheckerWithApproval,
    SafetyCheckResult,
)


class TestSafetyChecker:
    """SafetyChecker 基础测试."""

    def test_disabled_checker(self):
        """测试禁用的检查器."""
        checker = SafetyChecker(enabled=False)
        result = checker.check("bash", "rm -rf /")
        assert result.approved is True

    def test_dangerous_command_blocked(self):
        """测试危险命令被阻止."""
        checker = SafetyChecker()
        result = checker.check("bash", "rm -rf /")
        assert result.approved is False
        assert "Dangerous" in result.reason

    def test_sudo_blocked(self):
        """测试 sudo 被阻止."""
        checker = SafetyChecker()
        result = checker.check("bash", "sudo apt install")
        assert result.approved is False
        assert "sudo" in result.reason.lower()

    def test_safe_command_allowed(self):
        """测试安全命令允许."""
        checker = SafetyChecker()
        result = checker.check("bash", "ls -la")
        assert result.approved is True

    def test_protected_path_write_blocked(self):
        """测试受保护路径写入被阻止."""
        checker = SafetyChecker()
        result = checker.check("write", "/etc/passwd")
        assert result.approved is False

    def test_tmp_path_allowed(self):
        """测试 /tmp 路径允许."""
        checker = SafetyChecker()
        result = checker.check("write", "/tmp/test.txt")
        assert result.approved is True

    def test_add_protected_path(self):
        """测试添加受保护路径."""
        checker = SafetyChecker()
        checker.add_protected_path("/my/custom/path")
        result = checker.check("write", "/my/custom/path/file.txt")
        assert result.approved is False

    def test_add_dangerous_pattern(self):
        """测试添加危险模式."""
        checker = SafetyChecker()
        checker.add_dangerous_pattern(r"my_dangerous_command")
        result = checker.check("bash", "my_dangerous_command --all")
        assert result.approved is False


class TestSafetyCheckResult:
    """SafetyCheckResult 测试."""

    def test_basic_result(self):
        """测试基本结果."""
        result = SafetyCheckResult(approved=True)
        assert result.approved is True
        assert result.reason == ""
        assert result.modified_command is None

    def test_result_with_approval_request(self):
        """测试带审批请求的结果."""
        result = SafetyCheckResult(
            approved=False,
            reason="Requires approval",
            requires_approval=True,
            approval_request_id="req123",
        )
        assert result.requires_approval is True
        assert result.approval_request_id == "req123"


class TestApprovalRequest:
    """ApprovalRequest 测试."""

    def test_to_dict(self):
        """测试转换为字典."""
        request = ApprovalRequest(
            id="req1",
            command_type="bash",
            command="docker run",
            args={},
            worker_id="w1",
            task_id="t1",
            reason="Needs approval",
        )

        data = request.to_dict()
        assert data["id"] == "req1"
        assert data["command_type"] == "bash"
        assert data["status"] == "pending"
        assert "created_at" in data


class TestApprovalManager:
    """ApprovalManager 测试."""

    @pytest.mark.asyncio
    async def test_approve_request(self):
        """测试批准请求."""
        manager = ApprovalManager(timeout_seconds=5)

        # 在后台创建请求
        async def create_and_wait():
            return await manager.request_approval(
                command_type="bash",
                command="test",
                args={},
                worker_id="w1",
                task_id="t1",
                reason="test",
            )

        # 启动请求
        task = asyncio.create_task(create_and_wait())

        # 等待请求被创建
        await asyncio.sleep(0.1)

        # 获取待处理请求
        pending = manager.get_pending_requests()
        assert len(pending) == 1

        # 批准请求
        request_id = pending[0].id
        success = await manager.approve(request_id, approved_by="admin", comment="OK")
        assert success is True

        # 等待结果
        result = await task
        assert result.status == ApprovalStatus.APPROVED
        assert result.resolved_by == "admin"

    @pytest.mark.asyncio
    async def test_reject_request(self):
        """测试拒绝请求."""
        manager = ApprovalManager(timeout_seconds=5)

        async def create_and_wait():
            return await manager.request_approval(
                command_type="bash",
                command="test",
                args={},
                worker_id="w1",
                task_id="t1",
                reason="test",
            )

        task = asyncio.create_task(create_and_wait())
        await asyncio.sleep(0.1)

        pending = manager.get_pending_requests()
        request_id = pending[0].id
        success = await manager.reject(request_id, rejected_by="admin", comment="No")
        assert success is True

        result = await task
        assert result.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_timeout(self):
        """测试超时."""
        manager = ApprovalManager(timeout_seconds=0.1)

        result = await manager.request_approval(
            command_type="bash",
            command="test",
            args={},
            worker_id="w1",
            task_id="t1",
            reason="test",
        )

        assert result.status == ApprovalStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_notify_callback(self):
        """测试通知回调."""
        notifications = []

        def callback(request):
            notifications.append(request)

        manager = ApprovalManager(timeout_seconds=0.1, notify_callback=callback)

        await manager.request_approval(
            command_type="bash",
            command="test",
            args={},
            worker_id="w1",
            task_id="t1",
            reason="test",
        )

        assert len(notifications) == 1
        assert notifications[0].command == "test"


class TestSafetyCheckerWithApproval:
    """SafetyCheckerWithApproval 测试."""

    def test_approval_required_patterns(self):
        """测试需要审批的模式."""
        checker = SafetyCheckerWithApproval()

        # Docker run 需要审批
        result = checker.check("bash", "docker run nginx")
        assert result.approved is False
        assert result.requires_approval is True

        # Git force push 需要审批
        result = checker.check("bash", "git push --force origin main")
        assert result.approved is False
        assert result.requires_approval is True

        # 普通命令不需要审批
        result = checker.check("bash", "ls -la")
        assert result.approved is True
        assert result.requires_approval is False

    def test_dangerous_still_blocked(self):
        """测试危险命令仍被阻止."""
        checker = SafetyCheckerWithApproval()

        # rm -rf / 应该直接被阻止，不是审批
        result = checker.check("bash", "rm -rf /")
        assert result.approved is False
        assert result.requires_approval is False

    @pytest.mark.asyncio
    async def test_check_with_approval_approved(self):
        """测试带审批的检查 - 批准."""
        approval_manager = ApprovalManager(timeout_seconds=5)
        checker = SafetyCheckerWithApproval(approval_manager=approval_manager)

        async def check_and_wait():
            return await checker.check_with_approval(
                command_type="bash",
                command="docker run nginx",
                worker_id="w1",
                task_id="t1",
            )

        task = asyncio.create_task(check_and_wait())
        await asyncio.sleep(0.1)

        pending = approval_manager.get_pending_requests()
        await approval_manager.approve(pending[0].id)

        result = await task
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_check_with_approval_rejected(self):
        """测试带审批的检查 - 拒绝."""
        approval_manager = ApprovalManager(timeout_seconds=5)
        checker = SafetyCheckerWithApproval(approval_manager=approval_manager)

        async def check_and_wait():
            return await checker.check_with_approval(
                command_type="bash",
                command="docker run nginx",
                worker_id="w1",
                task_id="t1",
            )

        task = asyncio.create_task(check_and_wait())
        await asyncio.sleep(0.1)

        pending = approval_manager.get_pending_requests()
        await approval_manager.reject(pending[0].id, comment="Not allowed")

        result = await task
        assert result.approved is False
        assert "Rejected" in result.reason

    def test_add_approval_pattern(self):
        """测试添加审批模式."""
        checker = SafetyCheckerWithApproval()
        checker.add_approval_pattern(r"my_custom_command")

        result = checker.check("bash", "my_custom_command --arg")
        assert result.requires_approval is True
