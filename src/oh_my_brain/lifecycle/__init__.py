"""开发生命周期管理.

提供完整的开发生命周期支持，包括：
1. 项目阶段管理
2. 状态机驱动的流程控制
3. 检查点和回滚
4. 知识积累反馈
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProjectPhase(str, Enum):
    """项目阶段."""

    # 规划阶段
    INIT = "init"                     # 初始化
    REQUIREMENTS = "requirements"      # 需求分析
    DESIGN = "design"                  # 设计阶段

    # 开发阶段
    DEVELOPMENT = "development"        # 开发中
    CODE_REVIEW = "code_review"        # 代码审查
    TESTING = "testing"                # 测试阶段

    # 发布阶段
    STAGING = "staging"                # 预发布
    RELEASE = "release"                # 发布
    MAINTENANCE = "maintenance"        # 维护

    # 特殊状态
    PAUSED = "paused"                  # 暂停
    BLOCKED = "blocked"                # 阻塞
    COMPLETED = "completed"            # 完成
    CANCELLED = "cancelled"            # 取消


class TaskLifecycleState(str, Enum):
    """任务生命周期状态."""

    PENDING = "pending"           # 待处理
    QUEUED = "queued"            # 已排队
    ASSIGNED = "assigned"        # 已分配
    IN_PROGRESS = "in_progress"  # 进行中
    REVIEW = "review"            # 审查中
    TESTING = "testing"          # 测试中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    BLOCKED = "blocked"          # 阻塞
    CANCELLED = "cancelled"      # 已取消


class PhaseTransition(BaseModel):
    """阶段转换记录."""

    from_phase: ProjectPhase
    to_phase: ProjectPhase
    timestamp: datetime = Field(default_factory=datetime.now)
    reason: str = ""
    triggered_by: str = ""  # user, system, auto
    metadata: dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """检查点."""

    id: str
    phase: ProjectPhase
    timestamp: datetime = Field(default_factory=datetime.now)
    description: str = ""

    # 状态快照
    completed_modules: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)

    # 统计信息
    progress_percent: float = 0.0
    total_time_minutes: int = 0
    knowledge_entries_added: int = 0

    # 文件快照
    modified_files: list[str] = Field(default_factory=list)
    git_commit: str = ""

    # 可恢复性
    restorable: bool = True
    restore_instructions: str = ""


class LifecycleEvent(BaseModel):
    """生命周期事件."""

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    phase: ProjectPhase
    task_id: str | None = None
    module_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


# 阶段转换规则
PHASE_TRANSITIONS: dict[ProjectPhase, list[ProjectPhase]] = {
    ProjectPhase.INIT: [ProjectPhase.REQUIREMENTS, ProjectPhase.CANCELLED],
    ProjectPhase.REQUIREMENTS: [ProjectPhase.DESIGN, ProjectPhase.PAUSED, ProjectPhase.CANCELLED],
    ProjectPhase.DESIGN: [ProjectPhase.DEVELOPMENT, ProjectPhase.REQUIREMENTS, ProjectPhase.PAUSED],
    ProjectPhase.DEVELOPMENT: [ProjectPhase.CODE_REVIEW, ProjectPhase.TESTING, ProjectPhase.PAUSED, ProjectPhase.BLOCKED],
    ProjectPhase.CODE_REVIEW: [ProjectPhase.DEVELOPMENT, ProjectPhase.TESTING, ProjectPhase.PAUSED],
    ProjectPhase.TESTING: [ProjectPhase.DEVELOPMENT, ProjectPhase.STAGING, ProjectPhase.PAUSED],
    ProjectPhase.STAGING: [ProjectPhase.RELEASE, ProjectPhase.TESTING, ProjectPhase.PAUSED],
    ProjectPhase.RELEASE: [ProjectPhase.MAINTENANCE, ProjectPhase.COMPLETED],
    ProjectPhase.MAINTENANCE: [ProjectPhase.DEVELOPMENT, ProjectPhase.COMPLETED],
    ProjectPhase.PAUSED: [ProjectPhase.DEVELOPMENT, ProjectPhase.TESTING, ProjectPhase.CANCELLED],
    ProjectPhase.BLOCKED: [ProjectPhase.DEVELOPMENT, ProjectPhase.CANCELLED],
    ProjectPhase.COMPLETED: [],  # 终态
    ProjectPhase.CANCELLED: [],  # 终态
}


class LifecycleStateMachine:
    """生命周期状态机.

    管理项目阶段转换和事件处理。
    """

    def __init__(self, initial_phase: ProjectPhase = ProjectPhase.INIT):
        """初始化.

        Args:
            initial_phase: 初始阶段
        """
        self._current_phase = initial_phase
        self._transitions: list[PhaseTransition] = []
        self._event_handlers: dict[str, list[Callable]] = {}
        self._phase_entry_hooks: dict[ProjectPhase, list[Callable]] = {}
        self._phase_exit_hooks: dict[ProjectPhase, list[Callable]] = {}

    @property
    def current_phase(self) -> ProjectPhase:
        """当前阶段."""
        return self._current_phase

    @property
    def transition_history(self) -> list[PhaseTransition]:
        """转换历史."""
        return list(self._transitions)

    def can_transition_to(self, target_phase: ProjectPhase) -> bool:
        """检查是否可以转换到目标阶段.

        Args:
            target_phase: 目标阶段

        Returns:
            是否可以转换
        """
        allowed = PHASE_TRANSITIONS.get(self._current_phase, [])
        return target_phase in allowed

    def get_available_transitions(self) -> list[ProjectPhase]:
        """获取可用的转换目标.

        Returns:
            可转换的阶段列表
        """
        return PHASE_TRANSITIONS.get(self._current_phase, [])

    async def transition_to(
        self,
        target_phase: ProjectPhase,
        reason: str = "",
        triggered_by: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> PhaseTransition:
        """转换到目标阶段.

        Args:
            target_phase: 目标阶段
            reason: 转换原因
            triggered_by: 触发者
            metadata: 元数据

        Returns:
            转换记录

        Raises:
            ValueError: 不允许的转换
        """
        if not self.can_transition_to(target_phase):
            raise ValueError(
                f"不允许从 {self._current_phase.value} 转换到 {target_phase.value}. "
                f"可用转换: {[p.value for p in self.get_available_transitions()]}"
            )

        # 执行退出钩子
        await self._run_exit_hooks(self._current_phase)

        # 记录转换
        transition = PhaseTransition(
            from_phase=self._current_phase,
            to_phase=target_phase,
            reason=reason,
            triggered_by=triggered_by,
            metadata=metadata or {},
        )
        self._transitions.append(transition)

        # 更新状态
        old_phase = self._current_phase
        self._current_phase = target_phase

        # 执行进入钩子
        await self._run_entry_hooks(target_phase)

        # 触发事件
        await self._emit_event("phase_changed", {
            "from_phase": old_phase.value,
            "to_phase": target_phase.value,
            "reason": reason,
        })

        logger.info(f"阶段转换: {old_phase.value} -> {target_phase.value}")
        return transition

    def register_event_handler(
        self,
        event_type: str,
        handler: Callable,
    ) -> None:
        """注册事件处理器.

        Args:
            event_type: 事件类型
            handler: 处理器函数
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def register_phase_entry_hook(
        self,
        phase: ProjectPhase,
        hook: Callable,
    ) -> None:
        """注册阶段进入钩子.

        Args:
            phase: 阶段
            hook: 钩子函数
        """
        if phase not in self._phase_entry_hooks:
            self._phase_entry_hooks[phase] = []
        self._phase_entry_hooks[phase].append(hook)

    def register_phase_exit_hook(
        self,
        phase: ProjectPhase,
        hook: Callable,
    ) -> None:
        """注册阶段退出钩子.

        Args:
            phase: 阶段
            hook: 钩子函数
        """
        if phase not in self._phase_exit_hooks:
            self._phase_exit_hooks[phase] = []
        self._phase_exit_hooks[phase].append(hook)

    async def _run_entry_hooks(self, phase: ProjectPhase) -> None:
        """运行进入钩子."""
        hooks = self._phase_entry_hooks.get(phase, [])
        for hook in hooks:
            try:
                result = hook(phase)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error(f"阶段进入钩子执行失败: {e}")

    async def _run_exit_hooks(self, phase: ProjectPhase) -> None:
        """运行退出钩子."""
        hooks = self._phase_exit_hooks.get(phase, [])
        for hook in hooks:
            try:
                result = hook(phase)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error(f"阶段退出钩子执行失败: {e}")

    async def _emit_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """触发事件."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(event_type, data)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error(f"事件处理器执行失败: {e}")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "current_phase": self._current_phase.value,
            "transitions": [
                {
                    "from_phase": t.from_phase.value,
                    "to_phase": t.to_phase.value,
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason,
                    "triggered_by": t.triggered_by,
                }
                for t in self._transitions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LifecycleStateMachine":
        """从字典创建."""
        sm = cls(initial_phase=ProjectPhase(data["current_phase"]))
        for t_data in data.get("transitions", []):
            transition = PhaseTransition(
                from_phase=ProjectPhase(t_data["from_phase"]),
                to_phase=ProjectPhase(t_data["to_phase"]),
                timestamp=datetime.fromisoformat(t_data["timestamp"]),
                reason=t_data.get("reason", ""),
                triggered_by=t_data.get("triggered_by", ""),
            )
            sm._transitions.append(transition)
        return sm


class LifecycleManager:
    """开发生命周期管理器.

    统一管理项目的完整生命周期。
    """

    def __init__(
        self,
        project_name: str,
        persist_dir: Path | None = None,
    ):
        """初始化.

        Args:
            project_name: 项目名称
            persist_dir: 持久化目录
        """
        self._project_name = project_name
        self._persist_dir = persist_dir or Path.cwd() / ".oh_my_brain"
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._state_machine = LifecycleStateMachine()
        self._checkpoints: list[Checkpoint] = []
        self._events: list[LifecycleEvent] = []

        # 任务状态
        self._task_states: dict[str, TaskLifecycleState] = {}
        self._module_progress: dict[str, float] = {}

        # 统计
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

        # 加载持久化数据
        self._load()

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def current_phase(self) -> ProjectPhase:
        return self._state_machine.current_phase

    @property
    def is_active(self) -> bool:
        """项目是否处于活跃状态."""
        return self.current_phase not in [
            ProjectPhase.COMPLETED,
            ProjectPhase.CANCELLED,
            ProjectPhase.PAUSED,
        ]

    # ========== 阶段管理 ==========

    async def start_project(self) -> None:
        """启动项目."""
        self._start_time = datetime.now()
        await self._state_machine.transition_to(
            ProjectPhase.REQUIREMENTS,
            reason="项目启动",
            triggered_by="system",
        )
        self._record_event("project_started", {})
        self._save()

    async def advance_phase(
        self,
        target_phase: ProjectPhase | None = None,
        reason: str = "",
    ) -> ProjectPhase:
        """推进到下一阶段.

        Args:
            target_phase: 目标阶段（None 则自动选择）
            reason: 原因

        Returns:
            新阶段
        """
        if target_phase is None:
            # 自动选择下一阶段
            available = self._state_machine.get_available_transitions()
            # 优先选择非暂停/取消的阶段
            for phase in available:
                if phase not in [ProjectPhase.PAUSED, ProjectPhase.CANCELLED, ProjectPhase.BLOCKED]:
                    target_phase = phase
                    break

            if target_phase is None and available:
                target_phase = available[0]

        if target_phase is None:
            raise ValueError("没有可用的阶段转换")

        await self._state_machine.transition_to(
            target_phase,
            reason=reason,
            triggered_by="user",
        )

        self._record_event("phase_advanced", {"new_phase": target_phase.value})
        self._save()
        return target_phase

    async def pause_project(self, reason: str = "") -> None:
        """暂停项目."""
        await self._state_machine.transition_to(
            ProjectPhase.PAUSED,
            reason=reason or "用户暂停",
            triggered_by="user",
        )
        self._record_event("project_paused", {"reason": reason})
        self._save()

    async def resume_project(self) -> None:
        """恢复项目."""
        # 找到暂停前的阶段
        if self._state_machine.transition_history:
            last_active = None
            for t in reversed(self._state_machine.transition_history):
                if t.from_phase not in [ProjectPhase.PAUSED, ProjectPhase.BLOCKED]:
                    last_active = t.from_phase
                    break

            if last_active:
                await self._state_machine.transition_to(
                    last_active,
                    reason="恢复项目",
                    triggered_by="user",
                )
                self._record_event("project_resumed", {})
                self._save()

    async def complete_project(self) -> None:
        """完成项目."""
        self._end_time = datetime.now()

        # 先转到发布阶段
        if self.current_phase != ProjectPhase.RELEASE:
            if self._state_machine.can_transition_to(ProjectPhase.RELEASE):
                await self._state_machine.transition_to(
                    ProjectPhase.RELEASE,
                    reason="准备完成",
                    triggered_by="system",
                )

        # 转到完成状态
        await self._state_machine.transition_to(
            ProjectPhase.COMPLETED,
            reason="项目完成",
            triggered_by="user",
        )

        self._record_event("project_completed", {
            "duration_minutes": self._calculate_duration(),
        })
        self._save()

    # ========== 任务状态管理 ==========

    def update_task_state(
        self,
        task_id: str,
        new_state: TaskLifecycleState,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """更新任务状态.

        Args:
            task_id: 任务 ID
            new_state: 新状态
            metadata: 元数据
        """
        old_state = self._task_states.get(task_id)
        self._task_states[task_id] = new_state

        self._record_event("task_state_changed", {
            "task_id": task_id,
            "old_state": old_state.value if old_state else None,
            "new_state": new_state.value,
            **(metadata or {}),
        })

        # 检查是否需要自动阶段转换
        self._check_auto_phase_transition()
        self._save()

    def get_task_state(self, task_id: str) -> TaskLifecycleState | None:
        """获取任务状态."""
        return self._task_states.get(task_id)

    def update_module_progress(
        self,
        module_id: str,
        progress: float,
    ) -> None:
        """更新模块进度.

        Args:
            module_id: 模块 ID
            progress: 进度 (0-1)
        """
        self._module_progress[module_id] = min(1.0, max(0.0, progress))
        self._record_event("module_progress_updated", {
            "module_id": module_id,
            "progress": progress,
        })
        self._save()

    def get_overall_progress(self) -> float:
        """获取总体进度."""
        if not self._module_progress:
            return 0.0
        return sum(self._module_progress.values()) / len(self._module_progress)

    # ========== 检查点管理 ==========

    async def create_checkpoint(
        self,
        description: str = "",
        git_commit: str = "",
    ) -> Checkpoint:
        """创建检查点.

        Args:
            description: 描述
            git_commit: Git 提交 hash

        Returns:
            检查点
        """
        import uuid

        # 收集任务状态
        completed = [
            tid for tid, state in self._task_states.items()
            if state == TaskLifecycleState.COMPLETED
        ]
        pending = [
            tid for tid, state in self._task_states.items()
            if state in [TaskLifecycleState.PENDING, TaskLifecycleState.QUEUED]
        ]
        completed_modules = [
            mid for mid, prog in self._module_progress.items()
            if prog >= 1.0
        ]

        checkpoint = Checkpoint(
            id=f"cp-{uuid.uuid4().hex[:8]}",
            phase=self.current_phase,
            description=description or f"检查点 @ {self.current_phase.value}",
            completed_modules=completed_modules,
            completed_tasks=completed,
            pending_tasks=pending,
            progress_percent=self.get_overall_progress() * 100,
            total_time_minutes=self._calculate_duration(),
            git_commit=git_commit,
        )

        self._checkpoints.append(checkpoint)
        self._record_event("checkpoint_created", {"checkpoint_id": checkpoint.id})
        self._save()

        logger.info(f"创建检查点: {checkpoint.id}")
        return checkpoint

    def get_checkpoints(self) -> list[Checkpoint]:
        """获取所有检查点."""
        return list(self._checkpoints)

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """获取检查点."""
        for cp in self._checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """恢复到检查点.

        Args:
            checkpoint_id: 检查点 ID

        Returns:
            是否成功
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            logger.error(f"检查点不存在: {checkpoint_id}")
            return False

        if not checkpoint.restorable:
            logger.error(f"检查点不可恢复: {checkpoint_id}")
            return False

        # 恢复阶段
        if self._state_machine.can_transition_to(checkpoint.phase):
            await self._state_machine.transition_to(
                checkpoint.phase,
                reason=f"恢复到检查点 {checkpoint_id}",
                triggered_by="user",
            )

        # 恢复任务状态
        for task_id in checkpoint.completed_tasks:
            self._task_states[task_id] = TaskLifecycleState.COMPLETED
        for task_id in checkpoint.pending_tasks:
            self._task_states[task_id] = TaskLifecycleState.PENDING

        # 恢复模块进度
        for module_id in checkpoint.completed_modules:
            self._module_progress[module_id] = 1.0

        self._record_event("checkpoint_restored", {"checkpoint_id": checkpoint_id})
        self._save()

        logger.info(f"恢复到检查点: {checkpoint_id}")
        return True

    # ========== 事件和统计 ==========

    def _record_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """记录事件."""
        event = LifecycleEvent(
            event_type=event_type,
            phase=self.current_phase,
            data=data,
        )
        self._events.append(event)

        # 限制事件数量
        if len(self._events) > 1000:
            self._events = self._events[-500:]

    def get_events(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[LifecycleEvent]:
        """获取事件列表."""
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def _calculate_duration(self) -> int:
        """计算项目持续时间（分钟）."""
        if not self._start_time:
            return 0
        end = self._end_time or datetime.now()
        delta = end - self._start_time
        return int(delta.total_seconds() / 60)

    def _check_auto_phase_transition(self) -> None:
        """检查是否需要自动阶段转换."""
        # 如果所有任务完成且在开发阶段，自动进入测试
        if self.current_phase == ProjectPhase.DEVELOPMENT:
            all_completed = all(
                state == TaskLifecycleState.COMPLETED
                for state in self._task_states.values()
            )
            if all_completed and self._task_states:
                # 这里不自动转换，只记录建议
                self._record_event("auto_transition_suggested", {
                    "suggested_phase": ProjectPhase.TESTING.value,
                })

    def get_summary(self) -> dict[str, Any]:
        """获取项目摘要."""
        task_counts = {}
        for state in TaskLifecycleState:
            task_counts[state.value] = sum(
                1 for s in self._task_states.values() if s == state
            )

        return {
            "project_name": self._project_name,
            "current_phase": self.current_phase.value,
            "is_active": self.is_active,
            "overall_progress": round(self.get_overall_progress() * 100, 1),
            "duration_minutes": self._calculate_duration(),
            "task_counts": task_counts,
            "module_count": len(self._module_progress),
            "checkpoint_count": len(self._checkpoints),
            "transition_count": len(self._state_machine.transition_history),
        }

    # ========== 持久化 ==========

    def _save(self) -> None:
        """保存状态."""
        data = {
            "project_name": self._project_name,
            "state_machine": self._state_machine.to_dict(),
            "task_states": {k: v.value for k, v in self._task_states.items()},
            "module_progress": self._module_progress,
            "checkpoints": [cp.model_dump() for cp in self._checkpoints],
            "events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "phase": e.phase.value,
                    "data": e.data,
                }
                for e in self._events[-500:]  # 只保存最近500条
            ],
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
        }

        file_path = self._persist_dir / "lifecycle.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _load(self) -> None:
        """加载状态."""
        file_path = self._persist_dir / "lifecycle.json"
        if not file_path.exists():
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._project_name = data.get("project_name", self._project_name)
            self._state_machine = LifecycleStateMachine.from_dict(
                data.get("state_machine", {"current_phase": "init", "transitions": []})
            )
            self._task_states = {
                k: TaskLifecycleState(v)
                for k, v in data.get("task_states", {}).items()
            }
            self._module_progress = data.get("module_progress", {})

            # 加载检查点
            for cp_data in data.get("checkpoints", []):
                cp_data["timestamp"] = datetime.fromisoformat(cp_data["timestamp"])
                self._checkpoints.append(Checkpoint(**cp_data))

            # 加载事件
            for e_data in data.get("events", []):
                self._events.append(LifecycleEvent(
                    event_type=e_data["event_type"],
                    timestamp=datetime.fromisoformat(e_data["timestamp"]),
                    phase=ProjectPhase(e_data["phase"]),
                    data=e_data.get("data", {}),
                ))

            if data.get("start_time"):
                self._start_time = datetime.fromisoformat(data["start_time"])
            if data.get("end_time"):
                self._end_time = datetime.fromisoformat(data["end_time"])

            logger.info(f"加载生命周期状态: {self.current_phase.value}")

        except Exception as e:
            logger.error(f"加载生命周期状态失败: {e}")


def create_lifecycle_manager(
    project_name: str,
    project_dir: Path | None = None,
) -> LifecycleManager:
    """创建生命周期管理器.

    Args:
        project_name: 项目名称
        project_dir: 项目目录

    Returns:
        生命周期管理器
    """
    persist_dir = None
    if project_dir:
        persist_dir = Path(project_dir) / ".oh_my_brain"

    return LifecycleManager(
        project_name=project_name,
        persist_dir=persist_dir,
    )
