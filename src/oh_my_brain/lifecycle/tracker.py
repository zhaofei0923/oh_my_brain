"""进度追踪器.

提供详细的开发进度追踪和可视化。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from oh_my_brain.lifecycle import (
    LifecycleManager,
    ProjectPhase,
    TaskLifecycleState,
)

logger = logging.getLogger(__name__)


class ProgressMetric(str, Enum):
    """进度指标."""

    TASKS_COMPLETED = "tasks_completed"
    MODULES_COMPLETED = "modules_completed"
    CODE_COVERAGE = "code_coverage"
    BUGS_FIXED = "bugs_fixed"
    TIME_SPENT = "time_spent"
    VELOCITY = "velocity"


@dataclass
class TaskProgress:
    """任务进度."""

    task_id: str
    module_id: str
    description: str
    state: TaskLifecycleState
    assigned_worker: str | None = None

    # 时间追踪
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    estimated_minutes: int = 0
    actual_minutes: int = 0

    # 重试信息
    retry_count: int = 0
    last_error: str | None = None

    def is_overdue(self) -> bool:
        """是否超时."""
        if self.state == TaskLifecycleState.COMPLETED:
            return self.actual_minutes > self.estimated_minutes * 1.5
        if self.started_at and self.estimated_minutes:
            expected_end = self.started_at + timedelta(minutes=self.estimated_minutes)
            return datetime.now() > expected_end
        return False

    def get_efficiency(self) -> float | None:
        """获取效率（估计时间/实际时间）."""
        if self.actual_minutes > 0 and self.estimated_minutes > 0:
            return self.estimated_minutes / self.actual_minutes
        return None


@dataclass
class ModuleProgress:
    """模块进度."""

    module_id: str
    name: str
    priority: int = 2

    # 任务统计
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    in_progress_tasks: int = 0

    # 时间
    estimated_minutes: int = 0
    actual_minutes: int = 0

    # 依赖
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)  # 依赖此模块的其他模块

    @property
    def progress(self) -> float:
        """完成进度."""
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks

    @property
    def is_blocked(self) -> bool:
        """是否被阻塞."""
        return self.failed_tasks > 0 or self.total_tasks == 0


@dataclass
class PhaseProgress:
    """阶段进度."""

    phase: ProjectPhase
    status: str  # pending, active, completed, skipped

    # 时间
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_minutes: int = 0

    # 统计
    tasks_completed: int = 0
    issues_found: int = 0
    issues_resolved: int = 0


@dataclass
class ProgressSnapshot:
    """进度快照."""

    timestamp: datetime = field(default_factory=datetime.now)
    project_name: str = ""
    current_phase: ProjectPhase = ProjectPhase.INIT

    # 总体进度
    overall_progress: float = 0.0  # 0-1
    estimated_completion: datetime | None = None

    # 任务统计
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    in_progress_tasks: int = 0
    pending_tasks: int = 0

    # 模块统计
    total_modules: int = 0
    completed_modules: int = 0

    # 时间统计
    total_estimated_minutes: int = 0
    total_actual_minutes: int = 0
    elapsed_minutes: int = 0

    # 速度指标
    tasks_per_hour: float = 0.0
    average_task_minutes: float = 0.0

    # 健康指标
    retry_rate: float = 0.0  # 重试率
    success_rate: float = 0.0  # 成功率
    on_schedule: bool = True  # 是否按时


class ProgressTracker:
    """进度追踪器.

    功能：
    1. 实时进度追踪
    2. 速度计算
    3. 完成时间预估
    4. 健康状态检查
    """

    def __init__(self, lifecycle_manager: LifecycleManager):
        """初始化.

        Args:
            lifecycle_manager: 生命周期管理器
        """
        self._lifecycle = lifecycle_manager
        self._tasks: dict[str, TaskProgress] = {}
        self._modules: dict[str, ModuleProgress] = {}
        self._phases: dict[ProjectPhase, PhaseProgress] = {}
        self._snapshots: list[ProgressSnapshot] = []

        # 初始化阶段进度
        for phase in ProjectPhase:
            self._phases[phase] = PhaseProgress(phase=phase, status="pending")

    # ========== 任务追踪 ==========

    def register_task(
        self,
        task_id: str,
        module_id: str,
        description: str,
        estimated_minutes: int = 30,
    ) -> TaskProgress:
        """注册任务.

        Args:
            task_id: 任务 ID
            module_id: 模块 ID
            description: 描述
            estimated_minutes: 预估时间

        Returns:
            任务进度
        """
        task = TaskProgress(
            task_id=task_id,
            module_id=module_id,
            description=description,
            state=TaskLifecycleState.PENDING,
            created_at=datetime.now(),
            estimated_minutes=estimated_minutes,
        )
        self._tasks[task_id] = task

        # 更新模块统计
        if module_id in self._modules:
            self._modules[module_id].total_tasks += 1
            self._modules[module_id].estimated_minutes += estimated_minutes

        return task

    def start_task(
        self,
        task_id: str,
        worker_id: str | None = None,
    ) -> None:
        """开始任务.

        Args:
            task_id: 任务 ID
            worker_id: Worker ID
        """
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.state = TaskLifecycleState.IN_PROGRESS
        task.started_at = datetime.now()
        task.assigned_worker = worker_id

        # 更新模块统计
        if task.module_id in self._modules:
            self._modules[task.module_id].in_progress_tasks += 1

        # 更新生命周期
        self._lifecycle.update_task_state(task_id, TaskLifecycleState.IN_PROGRESS)

    def complete_task(
        self,
        task_id: str,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """完成任务.

        Args:
            task_id: 任务 ID
            success: 是否成功
            error: 错误消息
        """
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.completed_at = datetime.now()

        # 计算实际耗时
        if task.started_at:
            delta = task.completed_at - task.started_at
            task.actual_minutes = int(delta.total_seconds() / 60)

        if success:
            task.state = TaskLifecycleState.COMPLETED
        else:
            task.state = TaskLifecycleState.FAILED
            task.last_error = error
            task.retry_count += 1

        # 更新模块统计
        if task.module_id in self._modules:
            module = self._modules[task.module_id]
            module.in_progress_tasks = max(0, module.in_progress_tasks - 1)

            if success:
                module.completed_tasks += 1
                module.actual_minutes += task.actual_minutes
            else:
                module.failed_tasks += 1

        # 更新生命周期
        new_state = TaskLifecycleState.COMPLETED if success else TaskLifecycleState.FAILED
        self._lifecycle.update_task_state(task_id, new_state)

        # 更新模块进度
        self._update_module_progress(task.module_id)

    def retry_task(self, task_id: str) -> None:
        """重试任务.

        Args:
            task_id: 任务 ID
        """
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.state = TaskLifecycleState.PENDING
        task.started_at = None
        task.completed_at = None
        task.actual_minutes = 0

        self._lifecycle.update_task_state(task_id, TaskLifecycleState.PENDING)

    # ========== 模块追踪 ==========

    def register_module(
        self,
        module_id: str,
        name: str,
        priority: int = 2,
        dependencies: list[str] | None = None,
    ) -> ModuleProgress:
        """注册模块.

        Args:
            module_id: 模块 ID
            name: 名称
            priority: 优先级
            dependencies: 依赖列表

        Returns:
            模块进度
        """
        module = ModuleProgress(
            module_id=module_id,
            name=name,
            priority=priority,
            dependencies=dependencies or [],
        )
        self._modules[module_id] = module

        # 更新被依赖关系
        for dep_id in module.dependencies:
            if dep_id in self._modules:
                self._modules[dep_id].dependents.append(module_id)

        return module

    def _update_module_progress(self, module_id: str) -> None:
        """更新模块进度."""
        if module_id not in self._modules:
            return

        module = self._modules[module_id]
        self._lifecycle.update_module_progress(module_id, module.progress)

    def get_module_status(self, module_id: str) -> dict[str, Any]:
        """获取模块状态.

        Args:
            module_id: 模块 ID

        Returns:
            状态信息
        """
        if module_id not in self._modules:
            return {}

        module = self._modules[module_id]

        # 检查依赖是否完成
        deps_completed = all(
            self._modules.get(dep, ModuleProgress(module_id=dep, name="")).progress >= 1.0
            for dep in module.dependencies
        )

        return {
            "module_id": module_id,
            "name": module.name,
            "progress": round(module.progress * 100, 1),
            "total_tasks": module.total_tasks,
            "completed_tasks": module.completed_tasks,
            "failed_tasks": module.failed_tasks,
            "in_progress_tasks": module.in_progress_tasks,
            "dependencies_met": deps_completed,
            "is_blocked": module.is_blocked,
            "estimated_minutes": module.estimated_minutes,
            "actual_minutes": module.actual_minutes,
        }

    # ========== 阶段追踪 ==========

    def start_phase(self, phase: ProjectPhase) -> None:
        """开始阶段.

        Args:
            phase: 阶段
        """
        phase_progress = self._phases.get(phase)
        if phase_progress:
            phase_progress.status = "active"
            phase_progress.started_at = datetime.now()

    def complete_phase(self, phase: ProjectPhase) -> None:
        """完成阶段.

        Args:
            phase: 阶段
        """
        phase_progress = self._phases.get(phase)
        if phase_progress:
            phase_progress.status = "completed"
            phase_progress.completed_at = datetime.now()

            if phase_progress.started_at:
                delta = phase_progress.completed_at - phase_progress.started_at
                phase_progress.duration_minutes = int(delta.total_seconds() / 60)

    # ========== 统计和预估 ==========

    def take_snapshot(self) -> ProgressSnapshot:
        """获取当前进度快照.

        Returns:
            进度快照
        """
        snapshot = ProgressSnapshot(
            project_name=self._lifecycle.project_name,
            current_phase=self._lifecycle.current_phase,
        )

        # 任务统计
        for task in self._tasks.values():
            snapshot.total_tasks += 1
            snapshot.total_estimated_minutes += task.estimated_minutes
            snapshot.total_actual_minutes += task.actual_minutes

            if task.state == TaskLifecycleState.COMPLETED:
                snapshot.completed_tasks += 1
            elif task.state == TaskLifecycleState.FAILED:
                snapshot.failed_tasks += 1
            elif task.state == TaskLifecycleState.IN_PROGRESS:
                snapshot.in_progress_tasks += 1
            else:
                snapshot.pending_tasks += 1

        # 模块统计
        snapshot.total_modules = len(self._modules)
        snapshot.completed_modules = sum(
            1 for m in self._modules.values() if m.progress >= 1.0
        )

        # 计算总体进度
        if snapshot.total_tasks > 0:
            snapshot.overall_progress = snapshot.completed_tasks / snapshot.total_tasks

        # 计算速度
        if snapshot.total_actual_minutes > 0:
            snapshot.tasks_per_hour = (
                snapshot.completed_tasks / (snapshot.total_actual_minutes / 60)
            )
            if snapshot.completed_tasks > 0:
                snapshot.average_task_minutes = (
                    snapshot.total_actual_minutes / snapshot.completed_tasks
                )

        # 计算成功率和重试率
        finished = snapshot.completed_tasks + snapshot.failed_tasks
        if finished > 0:
            snapshot.success_rate = snapshot.completed_tasks / finished

        total_retries = sum(t.retry_count for t in self._tasks.values())
        if snapshot.total_tasks > 0:
            snapshot.retry_rate = total_retries / snapshot.total_tasks

        # 预估完成时间
        snapshot.estimated_completion = self._estimate_completion(snapshot)

        # 是否按时
        snapshot.on_schedule = (
            snapshot.total_actual_minutes <= snapshot.total_estimated_minutes * 1.2
        )

        self._snapshots.append(snapshot)
        return snapshot

    def _estimate_completion(self, snapshot: ProgressSnapshot) -> datetime | None:
        """预估完成时间."""
        if snapshot.completed_tasks == 0:
            return None

        if snapshot.pending_tasks + snapshot.in_progress_tasks == 0:
            return datetime.now()

        # 基于平均速度预估
        remaining_tasks = snapshot.pending_tasks + snapshot.in_progress_tasks
        if snapshot.average_task_minutes > 0:
            remaining_minutes = remaining_tasks * snapshot.average_task_minutes
            return datetime.now() + timedelta(minutes=remaining_minutes)

        return None

    def get_velocity(self, window_hours: int = 24) -> float:
        """获取最近的开发速度（任务/小时）.

        Args:
            window_hours: 时间窗口

        Returns:
            任务/小时
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_completed = sum(
            1 for t in self._tasks.values()
            if t.completed_at and t.completed_at > cutoff
            and t.state == TaskLifecycleState.COMPLETED
        )

        return recent_completed / window_hours

    # ========== 健康检查 ==========

    def get_health_status(self) -> dict[str, Any]:
        """获取健康状态.

        Returns:
            健康状态信息
        """
        snapshot = self.take_snapshot()

        issues = []
        warnings = []

        # 检查失败任务
        if snapshot.failed_tasks > 0:
            issues.append(f"{snapshot.failed_tasks} 个任务失败")

        # 检查重试率
        if snapshot.retry_rate > 0.2:
            warnings.append(f"重试率较高: {snapshot.retry_rate:.1%}")

        # 检查进度偏差
        if snapshot.total_actual_minutes > snapshot.total_estimated_minutes * 1.5:
            warnings.append("进度落后于预期")

        # 检查阻塞模块
        blocked_modules = [
            m.name for m in self._modules.values() if m.is_blocked
        ]
        if blocked_modules:
            issues.append(f"模块被阻塞: {', '.join(blocked_modules)}")

        # 检查超时任务
        overdue_tasks = [
            t.task_id for t in self._tasks.values() if t.is_overdue()
        ]
        if overdue_tasks:
            warnings.append(f"{len(overdue_tasks)} 个任务超时")

        status = "healthy"
        if issues:
            status = "critical"
        elif warnings:
            status = "warning"

        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "metrics": {
                "success_rate": round(snapshot.success_rate * 100, 1),
                "retry_rate": round(snapshot.retry_rate * 100, 1),
                "velocity": round(snapshot.tasks_per_hour, 2),
                "on_schedule": snapshot.on_schedule,
            },
        }

    # ========== 报告生成 ==========

    def generate_report(self) -> str:
        """生成进度报告.

        Returns:
            Markdown 格式报告
        """
        snapshot = self.take_snapshot()
        health = self.get_health_status()

        lines = [
            f"# 项目进度报告: {snapshot.project_name}",
            "",
            f"*生成时间: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## 总体状态",
            "",
            f"- **当前阶段**: {snapshot.current_phase.value}",
            f"- **总体进度**: {snapshot.overall_progress:.1%}",
            f"- **健康状态**: {health['status']}",
            "",
            "## 任务统计",
            "",
            f"| 状态 | 数量 |",
            f"|------|------|",
            f"| 已完成 | {snapshot.completed_tasks} |",
            f"| 进行中 | {snapshot.in_progress_tasks} |",
            f"| 待处理 | {snapshot.pending_tasks} |",
            f"| 失败 | {snapshot.failed_tasks} |",
            f"| **总计** | **{snapshot.total_tasks}** |",
            "",
            "## 模块进度",
            "",
            "| 模块 | 进度 | 任务 | 状态 |",
            "|------|------|------|------|",
        ]

        for module in sorted(self._modules.values(), key=lambda m: -m.priority):
            status = "✅" if module.progress >= 1.0 else ("🔴" if module.is_blocked else "🟡")
            lines.append(
                f"| {module.name} | {module.progress:.0%} | "
                f"{module.completed_tasks}/{module.total_tasks} | {status} |"
            )

        lines.extend([
            "",
            "## 时间统计",
            "",
            f"- **预估时间**: {snapshot.total_estimated_minutes} 分钟",
            f"- **实际耗时**: {snapshot.total_actual_minutes} 分钟",
            f"- **开发速度**: {snapshot.tasks_per_hour:.1f} 任务/小时",
        ])

        if snapshot.estimated_completion:
            lines.append(
                f"- **预计完成**: {snapshot.estimated_completion.strftime('%Y-%m-%d %H:%M')}"
            )

        if health["issues"]:
            lines.extend([
                "",
                "## ⚠️ 问题",
                "",
            ])
            for issue in health["issues"]:
                lines.append(f"- 🔴 {issue}")

        if health["warnings"]:
            lines.extend([
                "",
                "## ⚡ 警告",
                "",
            ])
            for warning in health["warnings"]:
                lines.append(f"- 🟡 {warning}")

        return "\n".join(lines)

    def get_gantt_data(self) -> list[dict[str, Any]]:
        """获取甘特图数据.

        Returns:
            甘特图数据列表
        """
        data = []

        for module in sorted(self._modules.values(), key=lambda m: m.priority):
            # 模块行
            module_tasks = [
                t for t in self._tasks.values()
                if t.module_id == module.module_id
            ]

            if not module_tasks:
                continue

            earliest_start = min(
                (t.started_at or t.created_at for t in module_tasks),
                default=datetime.now(),
            )
            latest_end = max(
                (t.completed_at or datetime.now() for t in module_tasks),
                default=datetime.now(),
            )

            data.append({
                "id": module.module_id,
                "name": module.name,
                "type": "module",
                "start": earliest_start.isoformat(),
                "end": latest_end.isoformat(),
                "progress": module.progress,
                "dependencies": module.dependencies,
            })

            # 任务行
            for task in module_tasks:
                data.append({
                    "id": task.task_id,
                    "name": task.description[:30],
                    "type": "task",
                    "parent": module.module_id,
                    "start": (task.started_at or task.created_at or datetime.now()).isoformat(),
                    "end": (task.completed_at or datetime.now()).isoformat(),
                    "progress": 1.0 if task.state == TaskLifecycleState.COMPLETED else 0.0,
                    "state": task.state.value,
                })

        return data
