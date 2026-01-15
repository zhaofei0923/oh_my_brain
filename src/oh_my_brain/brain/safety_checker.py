"""安全检查器.

对Worker执行的命令进行安全审核。
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """审批状态."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    """审批请求."""

    id: str
    command_type: str
    command: str
    args: dict[str, Any]
    worker_id: str
    task_id: str
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "id": self.id,
            "command_type": self.command_type,
            "command": self.command,
            "args": self.args,
            "worker_id": self.worker_id,
            "task_id": self.task_id,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "comment": self.comment,
        }


@dataclass
class SafetyCheckResult:
    """安全检查结果."""

    approved: bool
    reason: str = ""
    modified_command: str | None = None
    requires_approval: bool = False
    approval_request_id: str | None = None


class SafetyChecker:
    """安全检查器.

    检查Worker要执行的命令是否安全。

    检查类型：
    1. Bash命令 - 检查危险命令、路径安全
    2. 文件写入 - 检查目标路径是否安全
    3. 文件编辑 - 检查目标文件是否可编辑
    """

    # 默认危险命令模式
    DEFAULT_DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"rm\s+-rf\s+~",
        r"rm\s+-rf\s+\*",
        r"chmod\s+777",
        r"chmod\s+-R\s+777",
        r"curl\s+.*\|\s*bash",
        r"wget\s+.*\|\s*bash",
        r">\s*/dev/sd[a-z]",
        r"mkfs\.",
        r"dd\s+if=",
        r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",  # Fork bomb
        r"mv\s+/",
        r"cp\s+.*\s+/dev/",
        r"shutdown",
        r"reboot",
        r"init\s+[0-6]",
        r"systemctl\s+(stop|disable)\s+(docker|ssh|network)",
        r"iptables\s+-F",
        r"eval\s+\$",
        r"`.*`",  # 命令替换（可能危险）
    ]

    # 默认禁止写入的路径
    DEFAULT_PROTECTED_PATHS = [
        "/",
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/boot",
        "/lib",
        "/lib64",
        "/var",
        "/root",
        "/sys",
        "/proc",
        "/dev",
    ]

    def __init__(
        self,
        enabled: bool = True,
        dangerous_commands: list[str] | None = None,
        protected_paths: list[str] | None = None,
    ):
        self._enabled = enabled
        self._dangerous_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (dangerous_commands or self.DEFAULT_DANGEROUS_PATTERNS)
        ]
        self._protected_paths = protected_paths or self.DEFAULT_PROTECTED_PATHS

    def check(
        self,
        command_type: str,
        command: str,
        args: dict[str, Any] | None = None,
    ) -> SafetyCheckResult:
        """执行安全检查.

        Args:
            command_type: 命令类型 (bash, write, edit)
            command: 命令内容
            args: 额外参数

        Returns:
            检查结果
        """
        if not self._enabled:
            return SafetyCheckResult(approved=True)

        args = args or {}

        if command_type == "bash":
            return self._check_bash(command)
        elif command_type == "write":
            return self._check_write(command, args)
        elif command_type == "edit":
            return self._check_edit(command, args)
        else:
            # 未知类型，默认允许但记录警告
            logger.warning(f"Unknown command type: {command_type}")
            return SafetyCheckResult(approved=True)

    def _check_bash(self, command: str) -> SafetyCheckResult:
        """检查Bash命令."""
        # 检查危险模式
        for pattern in self._dangerous_patterns:
            if pattern.search(command):
                return SafetyCheckResult(
                    approved=False,
                    reason=f"Dangerous command pattern detected: {pattern.pattern}",
                )

        # 检查是否操作受保护路径
        for path in self._protected_paths:
            # 简单检查：命令中是否直接包含受保护路径
            if f" {path}" in command or f" {path}/" in command:
                # 允许读取，禁止写入/删除
                if any(
                    op in command for op in ["rm ", "mv ", "cp ", "> ", ">> ", "chmod ", "chown "]
                ):
                    return SafetyCheckResult(
                        approved=False,
                        reason=f"Operation on protected path: {path}",
                    )

        # 检查sudo
        if command.strip().startswith("sudo"):
            return SafetyCheckResult(
                approved=False,
                reason="sudo commands are not allowed",
            )

        return SafetyCheckResult(approved=True)

    def _check_write(self, file_path: str, args: dict) -> SafetyCheckResult:
        """检查文件写入."""
        # 检查目标路径
        for protected in self._protected_paths:
            if file_path.startswith(protected) and file_path != protected:
                # 允许写入 /tmp
                if file_path.startswith("/tmp"):
                    continue
                return SafetyCheckResult(
                    approved=False,
                    reason=f"Cannot write to protected path: {protected}",
                )

        # 检查危险文件扩展名
        dangerous_extensions = [".so", ".dll", ".exe", ".bin", ".sh"]
        for ext in dangerous_extensions:
            if file_path.endswith(ext):
                logger.warning(f"Writing to potentially dangerous file: {file_path}")
                # 允许但记录警告

        return SafetyCheckResult(approved=True)

    def _check_edit(self, file_path: str, args: dict) -> SafetyCheckResult:
        """检查文件编辑."""
        # 基本检查与write相同
        return self._check_write(file_path, args)

    def add_protected_path(self, path: str) -> None:
        """添加受保护路径."""
        if path not in self._protected_paths:
            self._protected_paths.append(path)

    def add_dangerous_pattern(self, pattern: str) -> None:
        """添加危险命令模式."""
        self._dangerous_patterns.append(re.compile(pattern, re.IGNORECASE))

    def is_enabled(self) -> bool:
        """检查是否启用."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """设置是否启用."""
        self._enabled = enabled


class ApprovalManager:
    """人工审批管理器.

    处理需要人工审批的操作，支持异步等待审批结果。
    """

    def __init__(
        self,
        timeout_seconds: int = 3600,  # 默认1小时超时
        notify_callback: Callable[[ApprovalRequest], None] | None = None,
    ):
        self._timeout_seconds = timeout_seconds
        self._notify_callback = notify_callback
        self._pending_requests: dict[str, ApprovalRequest] = {}
        self._approval_events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def request_approval(
        self,
        command_type: str,
        command: str,
        args: dict[str, Any],
        worker_id: str,
        task_id: str,
        reason: str,
    ) -> ApprovalRequest:
        """创建审批请求并等待结果.

        Args:
            command_type: 命令类型
            command: 命令内容
            args: 命令参数
            worker_id: 发起请求的 Worker ID
            task_id: 相关任务 ID
            reason: 需要审批的原因

        Returns:
            审批请求（包含最终状态）
        """
        request_id = str(uuid.uuid4())
        request = ApprovalRequest(
            id=request_id,
            command_type=command_type,
            command=command,
            args=args,
            worker_id=worker_id,
            task_id=task_id,
            reason=reason,
        )

        async with self._lock:
            self._pending_requests[request_id] = request
            self._approval_events[request_id] = asyncio.Event()

        # 通知回调（用于推送到 Dashboard 等）
        if self._notify_callback:
            try:
                self._notify_callback(request)
            except Exception as e:
                logger.error(f"Failed to notify approval request: {e}")

        logger.info(
            f"Approval request created: {request_id} for {command_type} "
            f"from worker {worker_id}"
        )

        # 等待审批结果或超时
        try:
            await asyncio.wait_for(
                self._approval_events[request_id].wait(),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            async with self._lock:
                if request_id in self._pending_requests:
                    request.status = ApprovalStatus.TIMEOUT
                    request.resolved_at = datetime.now()
                    del self._pending_requests[request_id]
            logger.warning(f"Approval request timed out: {request_id}")

        # 清理
        async with self._lock:
            if request_id in self._approval_events:
                del self._approval_events[request_id]

        return request

    async def approve(
        self,
        request_id: str,
        approved_by: str = "admin",
        comment: str = "",
    ) -> bool:
        """批准请求.

        Args:
            request_id: 请求 ID
            approved_by: 审批人
            comment: 备注

        Returns:
            是否成功
        """
        async with self._lock:
            if request_id not in self._pending_requests:
                return False

            request = self._pending_requests[request_id]
            request.status = ApprovalStatus.APPROVED
            request.resolved_at = datetime.now()
            request.resolved_by = approved_by
            request.comment = comment

            del self._pending_requests[request_id]

            if request_id in self._approval_events:
                self._approval_events[request_id].set()

        logger.info(f"Approval request approved: {request_id} by {approved_by}")
        return True

    async def reject(
        self,
        request_id: str,
        rejected_by: str = "admin",
        comment: str = "",
    ) -> bool:
        """拒绝请求.

        Args:
            request_id: 请求 ID
            rejected_by: 拒绝人
            comment: 拒绝原因

        Returns:
            是否成功
        """
        async with self._lock:
            if request_id not in self._pending_requests:
                return False

            request = self._pending_requests[request_id]
            request.status = ApprovalStatus.REJECTED
            request.resolved_at = datetime.now()
            request.resolved_by = rejected_by
            request.comment = comment

            del self._pending_requests[request_id]

            if request_id in self._approval_events:
                self._approval_events[request_id].set()

        logger.info(f"Approval request rejected: {request_id} by {rejected_by}")
        return True

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """获取所有待审批请求."""
        return list(self._pending_requests.values())

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """获取指定请求."""
        return self._pending_requests.get(request_id)

    def set_notify_callback(
        self, callback: Callable[[ApprovalRequest], None]
    ) -> None:
        """设置通知回调."""
        self._notify_callback = callback


class SafetyCheckerWithApproval(SafetyChecker):
    """带审批功能的安全检查器.

    扩展基础安全检查器，添加人工审批流程支持。
    对于部分危险但可能合法的操作，可以请求人工审批而非直接拒绝。
    """

    # 需要人工审批的模式（不是直接危险，但需要确认）
    APPROVAL_REQUIRED_PATTERNS = [
        r"pip\s+install\s+--upgrade",
        r"npm\s+install\s+-g",
        r"docker\s+run",
        r"docker\s+exec",
        r"git\s+push\s+--force",
        r"git\s+reset\s+--hard",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM",
        r"TRUNCATE\s+TABLE",
    ]

    def __init__(
        self,
        enabled: bool = True,
        dangerous_commands: list[str] | None = None,
        protected_paths: list[str] | None = None,
        approval_manager: ApprovalManager | None = None,
    ):
        super().__init__(enabled, dangerous_commands, protected_paths)
        self._approval_manager = approval_manager or ApprovalManager()
        self._approval_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.APPROVAL_REQUIRED_PATTERNS
        ]

    def check(
        self,
        command_type: str,
        command: str,
        args: dict[str, Any] | None = None,
    ) -> SafetyCheckResult:
        """执行安全检查.

        先进行基础安全检查，通过后再检查是否需要人工审批。
        """
        # 首先进行基础安全检查
        result = super().check(command_type, command, args)

        # 如果基础检查未通过，直接返回
        if not result.approved:
            return result

        # 检查是否需要人工审批
        for pattern in self._approval_patterns:
            if pattern.search(command):
                return SafetyCheckResult(
                    approved=False,
                    reason=f"This operation requires human approval: {pattern.pattern}",
                    requires_approval=True,
                )

        return result

    async def check_with_approval(
        self,
        command_type: str,
        command: str,
        args: dict[str, Any] | None = None,
        worker_id: str = "unknown",
        task_id: str = "unknown",
    ) -> SafetyCheckResult:
        """执行安全检查，如需要则等待人工审批.

        Args:
            command_type: 命令类型
            command: 命令内容
            args: 额外参数
            worker_id: Worker ID
            task_id: 任务 ID

        Returns:
            检查结果
        """
        result = self.check(command_type, command, args)

        if result.requires_approval and self._approval_manager:
            # 创建审批请求并等待
            approval = await self._approval_manager.request_approval(
                command_type=command_type,
                command=command,
                args=args or {},
                worker_id=worker_id,
                task_id=task_id,
                reason=result.reason,
            )

            if approval.status == ApprovalStatus.APPROVED:
                return SafetyCheckResult(
                    approved=True,
                    reason=f"Approved by {approval.resolved_by}: {approval.comment}",
                    approval_request_id=approval.id,
                )
            elif approval.status == ApprovalStatus.TIMEOUT:
                return SafetyCheckResult(
                    approved=False,
                    reason="Approval request timed out",
                    approval_request_id=approval.id,
                )
            else:
                return SafetyCheckResult(
                    approved=False,
                    reason=f"Rejected by {approval.resolved_by}: {approval.comment}",
                    approval_request_id=approval.id,
                )

        return result

    @property
    def approval_manager(self) -> ApprovalManager:
        """获取审批管理器."""
        return self._approval_manager

    def add_approval_pattern(self, pattern: str) -> None:
        """添加需要审批的模式."""
        self._approval_patterns.append(re.compile(pattern, re.IGNORECASE))
