"""安全检查器.

对Worker执行的命令进行安全审核。
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """安全检查结果."""

    approved: bool
    reason: str = ""
    modified_command: str | None = None


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
