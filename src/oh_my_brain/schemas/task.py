"""任务相关数据模型."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """任务类型（用于模型选择）."""

    PLANNING = "planning"  # 规划任务
    CODING = "coding"  # 编码任务
    REVIEW = "review"  # 审查任务
    TESTING = "testing"  # 测试任务
    DOCS = "docs"  # 文档任务
    REFACTOR = "refactor"  # 重构任务


class TaskStatus(str, Enum):
    """任务状态."""

    PENDING = "pending"  # 待分配
    QUEUED = "queued"  # 已入队，等待Worker
    ASSIGNED = "assigned"  # 已分配给Worker
    RUNNING = "running"  # 执行中
    PAUSED = "paused"  # 暂停
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class Task(BaseModel):
    """任务实体.

    表示一个可被Worker执行的任务单元。
    """

    id: str = Field(..., description="任务唯一ID")
    module_id: str = Field(..., description="所属模块ID")
    sub_task_id: str = Field(..., description="子任务ID（来自开发文档）")
    name: str = Field(..., description="任务名称")
    type: TaskType = Field(..., description="任务类型，用于选择AI模型")
    description: str = Field(..., description="任务描述")
    requirements: str = Field(..., description="具体需求")
    files_involved: list[str] = Field(default_factory=list, description="涉及文件")

    # 调度相关
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    priority: int = Field(3, ge=1, le=5, description="优先级")
    dependencies: list[str] = Field(default_factory=list, description="依赖的任务ID")
    estimated_minutes: int = Field(30, description="预估时长")

    # 执行相关
    assigned_worker: str | None = Field(None, description="分配的Worker ID")
    assigned_model: str | None = Field(None, description="使用的AI模型名称")
    git_branch: str | None = Field(None, description="Git工作分支")

    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: datetime | None = Field(None, description="开始执行时间")
    completed_at: datetime | None = Field(None, description="完成时间")

    # 进度
    progress: float = Field(0.0, ge=0.0, le=1.0, description="进度 0-1")
    checkpoint: dict[str, Any] | None = Field(None, description="检查点数据")

    model_config = {"extra": "forbid"}

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """检查任务是否就绪（所有依赖已完成）.

        Args:
            completed_tasks: 已完成的任务ID集合

        Returns:
            是否可以开始执行
        """
        return all(dep in completed_tasks for dep in self.dependencies)

    def mark_started(self, worker_id: str, model: str, branch: str) -> None:
        """标记任务开始执行."""
        self.status = TaskStatus.RUNNING
        self.assigned_worker = worker_id
        self.assigned_model = model
        self.git_branch = branch
        self.started_at = datetime.now()

    def mark_completed(self) -> None:
        """标记任务完成."""
        self.status = TaskStatus.COMPLETED
        self.progress = 1.0
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """标记任务失败."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()


class TaskResult(BaseModel):
    """任务执行结果."""

    task_id: str = Field(..., description="任务ID")
    success: bool = Field(..., description="是否成功")
    output: str = Field("", description="输出内容")
    error: str | None = Field(None, description="错误信息")
    files_modified: list[str] = Field(default_factory=list, description="修改的文件列表")
    git_commits: list[str] = Field(default_factory=list, description="提交的commit SHA")
    tokens_used: int = Field(0, description="消耗的token数量")
    duration_seconds: float = Field(0.0, description="执行耗时（秒）")
    model_used: str = Field("", description="使用的模型")

    # Worker上报的上下文更新
    context_update: dict[str, Any] | None = Field(None, description="上下文更新数据")

    model_config = {"extra": "forbid"}
