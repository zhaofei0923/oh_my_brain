"""阶段检查点.

提供阶段验证和检查点管理。
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from oh_my_brain.lifecycle import ProjectPhase

logger = logging.getLogger(__name__)


class CheckResult:
    """检查结果."""

    def __init__(
        self,
        passed: bool,
        message: str = "",
        details: dict[str, Any] | None = None,
    ):
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __bool__(self) -> bool:
        return self.passed

    @classmethod
    def success(cls, message: str = "检查通过") -> "CheckResult":
        return cls(passed=True, message=message)

    @classmethod
    def failure(cls, message: str, details: dict[str, Any] | None = None) -> "CheckResult":
        return cls(passed=False, message=message, details=details)


@dataclass
class CheckpointEntry:
    """检查点条目."""

    id: str
    name: str
    phase: ProjectPhase
    timestamp: datetime = field(default_factory=datetime.now)

    # 检查结果
    checks_passed: int = 0
    checks_failed: int = 0
    checks_skipped: int = 0

    # 状态快照
    state_hash: str = ""
    state_data: dict[str, Any] = field(default_factory=dict)

    # 元数据
    created_by: str = "system"
    notes: str = ""

    @property
    def total_checks(self) -> int:
        return self.checks_passed + self.checks_failed + self.checks_skipped

    @property
    def success_rate(self) -> float:
        executed = self.checks_passed + self.checks_failed
        if executed == 0:
            return 0.0
        return self.checks_passed / executed

    @property
    def is_valid(self) -> bool:
        return self.checks_failed == 0


class PhaseChecker(ABC):
    """阶段检查器基类."""

    @property
    @abstractmethod
    def phase(self) -> ProjectPhase:
        """适用阶段."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """检查器名称."""
        ...

    @abstractmethod
    def check(self, context: dict[str, Any]) -> CheckResult:
        """执行检查.

        Args:
            context: 检查上下文

        Returns:
            检查结果
        """
        ...


class RequirementsChecker(PhaseChecker):
    """需求检查器."""

    @property
    def phase(self) -> ProjectPhase:
        return ProjectPhase.REQUIREMENTS

    @property
    def name(self) -> str:
        return "requirements_checker"

    def check(self, context: dict[str, Any]) -> CheckResult:
        """检查需求是否完整."""
        requirements = context.get("requirements", [])
        dev_doc = context.get("dev_doc", {})

        issues = []

        # 检查是否有需求
        if not requirements:
            issues.append("未定义任何需求")

        # 检查开发文档
        if not dev_doc:
            issues.append("缺少开发文档")
        else:
            # 检查必要字段
            required_fields = ["project_name", "modules"]
            for field in required_fields:
                if field not in dev_doc:
                    issues.append(f"开发文档缺少字段: {field}")

        if issues:
            return CheckResult.failure(
                "需求检查失败",
                {"issues": issues},
            )

        return CheckResult.success("需求检查通过")


class DesignChecker(PhaseChecker):
    """设计检查器."""

    @property
    def phase(self) -> ProjectPhase:
        return ProjectPhase.DESIGN

    @property
    def name(self) -> str:
        return "design_checker"

    def check(self, context: dict[str, Any]) -> CheckResult:
        """检查设计是否完整."""
        dev_doc = context.get("dev_doc", {})
        issues = []

        # 检查模块设计
        modules = dev_doc.get("modules", [])
        if not modules:
            issues.append("未定义任何模块")
        else:
            for module in modules:
                if not module.get("name"):
                    issues.append(f"模块缺少名称: {module}")
                if not module.get("tasks"):
                    issues.append(f"模块 '{module.get('name', '?')}' 未定义任务")

        # 检查技术栈
        if not dev_doc.get("tech_stack"):
            issues.append("未定义技术栈")

        if issues:
            return CheckResult.failure("设计检查失败", {"issues": issues})

        return CheckResult.success("设计检查通过")


class ImplementationChecker(PhaseChecker):
    """实现检查器."""

    @property
    def phase(self) -> ProjectPhase:
        return ProjectPhase.IMPLEMENTATION

    @property
    def name(self) -> str:
        return "implementation_checker"

    def check(self, context: dict[str, Any]) -> CheckResult:
        """检查实现进度."""
        progress = context.get("progress", {})
        issues = []
        warnings = []

        total_tasks = progress.get("total_tasks", 0)
        completed_tasks = progress.get("completed_tasks", 0)
        failed_tasks = progress.get("failed_tasks", 0)

        if total_tasks == 0:
            issues.append("未定义任何任务")

        if failed_tasks > 0:
            issues.append(f"{failed_tasks} 个任务执行失败")

        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        if completion_rate < 0.5:
            warnings.append(f"完成率较低: {completion_rate:.1%}")

        if issues:
            return CheckResult.failure(
                "实现检查失败",
                {"issues": issues, "warnings": warnings},
            )

        return CheckResult.success(
            f"实现检查通过 ({completed_tasks}/{total_tasks} 任务完成)"
        )


class TestingChecker(PhaseChecker):
    """测试检查器."""

    @property
    def phase(self) -> ProjectPhase:
        return ProjectPhase.TESTING

    @property
    def name(self) -> str:
        return "testing_checker"

    def check(self, context: dict[str, Any]) -> CheckResult:
        """检查测试状态."""
        test_results = context.get("test_results", {})
        issues = []

        # 检查测试覆盖率
        coverage = test_results.get("coverage", 0)
        min_coverage = context.get("min_coverage", 80)

        if coverage < min_coverage:
            issues.append(f"测试覆盖率 ({coverage}%) 低于要求 ({min_coverage}%)")

        # 检查测试通过率
        total_tests = test_results.get("total", 0)
        passed_tests = test_results.get("passed", 0)
        failed_tests = test_results.get("failed", 0)

        if total_tests == 0:
            issues.append("未执行任何测试")
        elif failed_tests > 0:
            issues.append(f"{failed_tests} 个测试失败")

        if issues:
            return CheckResult.failure("测试检查失败", {"issues": issues})

        return CheckResult.success(
            f"测试检查通过 (覆盖率: {coverage}%, 通过: {passed_tests}/{total_tests})"
        )


class ReviewChecker(PhaseChecker):
    """审查检查器."""

    @property
    def phase(self) -> ProjectPhase:
        return ProjectPhase.REVIEW

    @property
    def name(self) -> str:
        return "review_checker"

    def check(self, context: dict[str, Any]) -> CheckResult:
        """检查审查状态."""
        reviews = context.get("reviews", [])
        issues = []

        if not reviews:
            issues.append("未进行任何审查")
            return CheckResult.failure("审查检查失败", {"issues": issues})

        # 检查是否有未解决的问题
        unresolved = sum(
            1 for r in reviews
            if r.get("status") == "open"
        )

        if unresolved > 0:
            issues.append(f"{unresolved} 个审查问题未解决")

        # 检查是否有批准
        approved = any(r.get("status") == "approved" for r in reviews)
        if not approved:
            issues.append("审查未获批准")

        if issues:
            return CheckResult.failure("审查检查失败", {"issues": issues})

        return CheckResult.success("审查检查通过")


class CheckpointManager:
    """检查点管理器.

    功能：
    1. 注册和管理检查器
    2. 执行阶段检查
    3. 创建和验证检查点
    4. 持久化检查点
    """

    # 默认检查器
    DEFAULT_CHECKERS: list[type[PhaseChecker]] = [
        RequirementsChecker,
        DesignChecker,
        ImplementationChecker,
        TestingChecker,
        ReviewChecker,
    ]

    def __init__(self, storage_path: Path | None = None):
        """初始化.

        Args:
            storage_path: 检查点存储路径
        """
        self._storage_path = storage_path or Path(".oh_my_brain/checkpoints")
        self._checkers: dict[ProjectPhase, list[PhaseChecker]] = {}
        self._checkpoints: list[CheckpointEntry] = []
        self._custom_checks: dict[ProjectPhase, list[Callable]] = {}

        # 注册默认检查器
        for checker_cls in self.DEFAULT_CHECKERS:
            self.register_checker(checker_cls())

    def register_checker(self, checker: PhaseChecker) -> None:
        """注册检查器.

        Args:
            checker: 检查器实例
        """
        phase = checker.phase
        if phase not in self._checkers:
            self._checkers[phase] = []
        self._checkers[phase].append(checker)
        logger.debug(f"注册检查器: {checker.name} -> {phase.value}")

    def add_custom_check(
        self,
        phase: ProjectPhase,
        check_func: Callable[[dict[str, Any]], CheckResult],
    ) -> None:
        """添加自定义检查.

        Args:
            phase: 阶段
            check_func: 检查函数
        """
        if phase not in self._custom_checks:
            self._custom_checks[phase] = []
        self._custom_checks[phase].append(check_func)

    def run_phase_checks(
        self,
        phase: ProjectPhase,
        context: dict[str, Any],
    ) -> tuple[list[CheckResult], list[CheckResult]]:
        """运行阶段检查.

        Args:
            phase: 阶段
            context: 检查上下文

        Returns:
            (通过的检查列表, 失败的检查列表)
        """
        passed = []
        failed = []

        # 运行注册的检查器
        checkers = self._checkers.get(phase, [])
        for checker in checkers:
            try:
                result = checker.check(context)
                if result:
                    passed.append(result)
                else:
                    failed.append(result)
            except Exception as e:
                logger.error(f"检查器 {checker.name} 执行出错: {e}")
                failed.append(CheckResult.failure(
                    f"检查器执行错误: {e}",
                    {"checker": checker.name, "error": str(e)},
                ))

        # 运行自定义检查
        custom_checks = self._custom_checks.get(phase, [])
        for check_func in custom_checks:
            try:
                result = check_func(context)
                if result:
                    passed.append(result)
                else:
                    failed.append(result)
            except Exception as e:
                logger.error(f"自定义检查执行出错: {e}")
                failed.append(CheckResult.failure(f"自定义检查错误: {e}"))

        return passed, failed

    def create_checkpoint(
        self,
        phase: ProjectPhase,
        context: dict[str, Any],
        run_checks: bool = True,
        notes: str = "",
    ) -> CheckpointEntry:
        """创建检查点.

        Args:
            phase: 阶段
            context: 上下文数据
            run_checks: 是否运行检查
            notes: 备注

        Returns:
            检查点条目
        """
        # 生成 ID
        timestamp = datetime.now()
        checkpoint_id = f"{phase.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # 计算状态哈希
        state_hash = self._compute_hash(context)

        entry = CheckpointEntry(
            id=checkpoint_id,
            name=f"{phase.value} 检查点",
            phase=phase,
            timestamp=timestamp,
            state_hash=state_hash,
            state_data=context.copy(),
            notes=notes,
        )

        # 运行检查
        if run_checks:
            passed, failed = self.run_phase_checks(phase, context)
            entry.checks_passed = len(passed)
            entry.checks_failed = len(failed)

        self._checkpoints.append(entry)
        self._save_checkpoint(entry)

        return entry

    def _compute_hash(self, data: dict[str, Any]) -> str:
        """计算数据哈希."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _save_checkpoint(self, entry: CheckpointEntry) -> None:
        """保存检查点到文件."""
        self._storage_path.mkdir(parents=True, exist_ok=True)
        file_path = self._storage_path / f"{entry.id}.json"

        data = {
            "id": entry.id,
            "name": entry.name,
            "phase": entry.phase.value,
            "timestamp": entry.timestamp.isoformat(),
            "checks_passed": entry.checks_passed,
            "checks_failed": entry.checks_failed,
            "checks_skipped": entry.checks_skipped,
            "state_hash": entry.state_hash,
            "state_data": entry.state_data,
            "created_by": entry.created_by,
            "notes": entry.notes,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointEntry | None:
        """加载检查点.

        Args:
            checkpoint_id: 检查点 ID

        Returns:
            检查点条目，不存在则返回 None
        """
        file_path = self._storage_path / f"{checkpoint_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        return CheckpointEntry(
            id=data["id"],
            name=data["name"],
            phase=ProjectPhase(data["phase"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            checks_passed=data["checks_passed"],
            checks_failed=data["checks_failed"],
            checks_skipped=data.get("checks_skipped", 0),
            state_hash=data["state_hash"],
            state_data=data.get("state_data", {}),
            created_by=data.get("created_by", "system"),
            notes=data.get("notes", ""),
        )

    def get_latest_checkpoint(
        self,
        phase: ProjectPhase | None = None,
    ) -> CheckpointEntry | None:
        """获取最新检查点.

        Args:
            phase: 可选，指定阶段

        Returns:
            最新检查点
        """
        checkpoints = self._checkpoints
        if phase:
            checkpoints = [c for c in checkpoints if c.phase == phase]

        if not checkpoints:
            return None

        return max(checkpoints, key=lambda c: c.timestamp)

    def verify_checkpoint(self, entry: CheckpointEntry) -> CheckResult:
        """验证检查点.

        Args:
            entry: 检查点条目

        Returns:
            验证结果
        """
        # 重新运行检查
        passed, failed = self.run_phase_checks(entry.phase, entry.state_data)

        if failed:
            return CheckResult.failure(
                f"检查点验证失败 ({len(failed)} 项检查未通过)",
                {"failed_checks": [f.message for f in failed]},
            )

        return CheckResult.success(f"检查点验证通过 ({len(passed)} 项检查)")

    def compare_checkpoints(
        self,
        checkpoint1: CheckpointEntry,
        checkpoint2: CheckpointEntry,
    ) -> dict[str, Any]:
        """比较两个检查点.

        Args:
            checkpoint1: 第一个检查点
            checkpoint2: 第二个检查点

        Returns:
            差异信息
        """
        return {
            "time_diff_minutes": (
                (checkpoint2.timestamp - checkpoint1.timestamp).total_seconds() / 60
            ),
            "hash_changed": checkpoint1.state_hash != checkpoint2.state_hash,
            "checks_diff": {
                "passed": checkpoint2.checks_passed - checkpoint1.checks_passed,
                "failed": checkpoint2.checks_failed - checkpoint1.checks_failed,
            },
            "phase_changed": checkpoint1.phase != checkpoint2.phase,
        }

    def list_checkpoints(
        self,
        phase: ProjectPhase | None = None,
        valid_only: bool = False,
    ) -> list[CheckpointEntry]:
        """列出检查点.

        Args:
            phase: 可选，过滤阶段
            valid_only: 是否只返回有效检查点

        Returns:
            检查点列表
        """
        result = self._checkpoints

        if phase:
            result = [c for c in result if c.phase == phase]

        if valid_only:
            result = [c for c in result if c.is_valid]

        return sorted(result, key=lambda c: c.timestamp, reverse=True)

    def cleanup_old_checkpoints(
        self,
        keep_count: int = 10,
        keep_valid: bool = True,
    ) -> int:
        """清理旧检查点.

        Args:
            keep_count: 每个阶段保留数量
            keep_valid: 是否保留所有有效检查点

        Returns:
            删除的检查点数量
        """
        deleted = 0

        # 按阶段分组
        by_phase: dict[ProjectPhase, list[CheckpointEntry]] = {}
        for checkpoint in self._checkpoints:
            if checkpoint.phase not in by_phase:
                by_phase[checkpoint.phase] = []
            by_phase[checkpoint.phase].append(checkpoint)

        # 清理每个阶段
        for phase, checkpoints in by_phase.items():
            # 按时间排序（新的在前）
            sorted_cps = sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)

            # 保留的检查点
            to_keep = set()

            # 保留最新的
            for cp in sorted_cps[:keep_count]:
                to_keep.add(cp.id)

            # 保留有效的
            if keep_valid:
                for cp in sorted_cps:
                    if cp.is_valid:
                        to_keep.add(cp.id)

            # 删除其他的
            for cp in sorted_cps:
                if cp.id not in to_keep:
                    file_path = self._storage_path / f"{cp.id}.json"
                    if file_path.exists():
                        file_path.unlink()
                    self._checkpoints.remove(cp)
                    deleted += 1

        return deleted

    def get_phase_gate_status(
        self,
        phase: ProjectPhase,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """获取阶段门禁状态.

        Args:
            phase: 阶段
            context: 上下文

        Returns:
            门禁状态
        """
        passed, failed = self.run_phase_checks(phase, context)

        return {
            "phase": phase.value,
            "can_proceed": len(failed) == 0,
            "checks": {
                "passed": len(passed),
                "failed": len(failed),
                "total": len(passed) + len(failed),
            },
            "failures": [
                {"message": f.message, "details": f.details}
                for f in failed
            ],
            "timestamp": datetime.now().isoformat(),
        }
