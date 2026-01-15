"""反馈循环集成.

将生命周期与知识库学习连接，实现持续改进。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from oh_my_brain.knowledge import (
    BugFixEntry,
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeType,
)
from oh_my_brain.lifecycle import (
    LifecycleEvent,
    LifecycleEventType,
    LifecycleManager,
    ProjectPhase,
    TaskLifecycleState,
)
from oh_my_brain.lifecycle.tracker import ProgressTracker, TaskProgress

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """反馈类型."""

    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    BUG_FIX = "bug_fix"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    USER = "user"
    SYSTEM = "system"


class InsightType(str, Enum):
    """洞察类型."""

    PATTERN = "pattern"
    BOTTLENECK = "bottleneck"
    IMPROVEMENT = "improvement"
    WARNING = "warning"
    BEST_PRACTICE = "best_practice"


@dataclass
class Feedback:
    """反馈条目."""

    id: str
    type: FeedbackType
    source: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    # 上下文
    phase: ProjectPhase | None = None
    task_id: str | None = None
    module_id: str | None = None

    # 元数据
    severity: str = "info"  # info, warning, error
    metadata: dict[str, Any] = field(default_factory=dict)

    # 处理状态
    processed: bool = False
    action_taken: str | None = None


@dataclass
class Insight:
    """洞察条目."""

    id: str
    type: InsightType
    title: str
    description: str
    confidence: float = 0.8  # 置信度

    # 来源
    source_feedbacks: list[str] = field(default_factory=list)
    supporting_data: dict[str, Any] = field(default_factory=dict)

    # 建议
    recommendations: list[str] = field(default_factory=list)

    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    applied: bool = False


class FeedbackCollector:
    """反馈收集器.

    收集来自各个阶段和组件的反馈。
    """

    def __init__(self):
        self._feedbacks: list[Feedback] = []
        self._feedback_handlers: dict[FeedbackType, list[Callable]] = {}
        self._feedback_counter = 0

    def collect(
        self,
        feedback_type: FeedbackType,
        source: str,
        content: str,
        **kwargs: Any,
    ) -> Feedback:
        """收集反馈.

        Args:
            feedback_type: 反馈类型
            source: 来源
            content: 内容
            **kwargs: 额外参数

        Returns:
            反馈条目
        """
        self._feedback_counter += 1
        feedback = Feedback(
            id=f"fb_{self._feedback_counter}",
            type=feedback_type,
            source=source,
            content=content,
            phase=kwargs.get("phase"),
            task_id=kwargs.get("task_id"),
            module_id=kwargs.get("module_id"),
            severity=kwargs.get("severity", "info"),
            metadata=kwargs.get("metadata", {}),
        )

        self._feedbacks.append(feedback)

        # 触发处理器
        handlers = self._feedback_handlers.get(feedback_type, [])
        for handler in handlers:
            try:
                handler(feedback)
            except Exception as e:
                logger.error(f"反馈处理器执行错误: {e}")

        return feedback

    def register_handler(
        self,
        feedback_type: FeedbackType,
        handler: Callable[[Feedback], None],
    ) -> None:
        """注册反馈处理器.

        Args:
            feedback_type: 反馈类型
            handler: 处理函数
        """
        if feedback_type not in self._feedback_handlers:
            self._feedback_handlers[feedback_type] = []
        self._feedback_handlers[feedback_type].append(handler)

    def get_feedbacks(
        self,
        feedback_type: FeedbackType | None = None,
        phase: ProjectPhase | None = None,
        unprocessed_only: bool = False,
    ) -> list[Feedback]:
        """获取反馈列表.

        Args:
            feedback_type: 过滤类型
            phase: 过滤阶段
            unprocessed_only: 只返回未处理的

        Returns:
            反馈列表
        """
        result = self._feedbacks

        if feedback_type:
            result = [f for f in result if f.type == feedback_type]

        if phase:
            result = [f for f in result if f.phase == phase]

        if unprocessed_only:
            result = [f for f in result if not f.processed]

        return result


class InsightGenerator:
    """洞察生成器.

    分析反馈数据，生成有价值的洞察。
    """

    def __init__(self):
        self._insights: list[Insight] = []
        self._insight_counter = 0

    def analyze_task_patterns(
        self,
        tasks: list[TaskProgress],
    ) -> list[Insight]:
        """分析任务模式.

        Args:
            tasks: 任务列表

        Returns:
            洞察列表
        """
        insights = []

        if not tasks:
            return insights

        # 分析失败模式
        failed_tasks = [t for t in tasks if t.state == TaskLifecycleState.FAILED]
        if len(failed_tasks) >= 3:
            # 检查是否有共同模式
            modules = [t.module_id for t in failed_tasks]
            most_common_module = max(set(modules), key=modules.count)
            if modules.count(most_common_module) >= 2:
                self._insight_counter += 1
                insights.append(Insight(
                    id=f"insight_{self._insight_counter}",
                    type=InsightType.PATTERN,
                    title="模块失败集中",
                    description=f"模块 '{most_common_module}' 存在多个失败任务",
                    confidence=0.9,
                    recommendations=[
                        f"检查模块 '{most_common_module}' 的设计和依赖",
                        "考虑重构或拆分该模块",
                    ],
                ))

        # 分析效率问题
        overdue_tasks = [t for t in tasks if t.is_overdue()]
        if len(overdue_tasks) >= 2:
            self._insight_counter += 1
            avg_overrun = sum(
                (t.actual_minutes - t.estimated_minutes)
                for t in overdue_tasks
                if t.actual_minutes > 0 and t.estimated_minutes > 0
            ) / len(overdue_tasks)

            insights.append(Insight(
                id=f"insight_{self._insight_counter}",
                type=InsightType.BOTTLENECK,
                title="时间估计偏差",
                description=f"多个任务超时，平均超出 {avg_overrun:.0f} 分钟",
                confidence=0.85,
                recommendations=[
                    "重新评估任务复杂度",
                    "考虑增加预估时间缓冲",
                    "检查是否有阻塞因素",
                ],
            ))

        # 分析高效模式
        efficient_tasks = [
            t for t in tasks
            if t.get_efficiency() and t.get_efficiency() > 1.2  # type: ignore
        ]
        if len(efficient_tasks) >= 2:
            self._insight_counter += 1
            insights.append(Insight(
                id=f"insight_{self._insight_counter}",
                type=InsightType.BEST_PRACTICE,
                title="高效任务模式",
                description=f"{len(efficient_tasks)} 个任务完成效率超过预期",
                confidence=0.8,
                supporting_data={
                    "efficient_tasks": [t.task_id for t in efficient_tasks],
                },
                recommendations=[
                    "分析这些任务的共同特点",
                    "将高效模式应用到其他任务",
                ],
            ))

        self._insights.extend(insights)
        return insights

    def analyze_phase_transitions(
        self,
        events: list[LifecycleEvent],
    ) -> list[Insight]:
        """分析阶段转换.

        Args:
            events: 事件列表

        Returns:
            洞察列表
        """
        insights = []

        # 查找阶段转换事件
        transition_events = [
            e for e in events
            if e.event_type == LifecycleEventType.PHASE_CHANGE
        ]

        if len(transition_events) < 2:
            return insights

        # 分析转换时间
        for i in range(1, len(transition_events)):
            prev = transition_events[i - 1]
            curr = transition_events[i]

            duration = (curr.timestamp - prev.timestamp).total_seconds() / 60

            # 如果阶段持续时间过长
            if duration > 120:  # 2 小时
                self._insight_counter += 1
                insights.append(Insight(
                    id=f"insight_{self._insight_counter}",
                    type=InsightType.WARNING,
                    title="阶段耗时较长",
                    description=(
                        f"阶段 '{prev.data.get('phase', '?')}' "
                        f"持续 {duration:.0f} 分钟"
                    ),
                    confidence=0.7,
                    recommendations=[
                        "检查该阶段是否存在阻塞",
                        "考虑拆分为更小的阶段",
                    ],
                ))

        self._insights.extend(insights)
        return insights


class FeedbackLoop:
    """反馈循环.

    连接生命周期管理器、进度追踪器和知识库。
    """

    def __init__(
        self,
        lifecycle_manager: LifecycleManager,
        tracker: ProgressTracker,
        knowledge_base: KnowledgeBase,
    ):
        """初始化.

        Args:
            lifecycle_manager: 生命周期管理器
            tracker: 进度追踪器
            knowledge_base: 知识库
        """
        self._lifecycle = lifecycle_manager
        self._tracker = tracker
        self._kb = knowledge_base

        self._collector = FeedbackCollector()
        self._insight_gen = InsightGenerator()

        # 设置事件监听
        self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """设置事件监听器."""
        # 监听任务完成
        self._lifecycle.on_event(
            LifecycleEventType.TASK_COMPLETED,
            self._on_task_completed,
        )

        # 监听任务失败
        self._lifecycle.on_event(
            LifecycleEventType.TASK_FAILED,
            self._on_task_failed,
        )

        # 监听阶段变更
        self._lifecycle.on_event(
            LifecycleEventType.PHASE_CHANGE,
            self._on_phase_change,
        )

    def _on_task_completed(self, event: LifecycleEvent) -> None:
        """任务完成处理."""
        task_id = event.data.get("task_id", "")
        task = self._tracker._tasks.get(task_id)

        if not task:
            return

        # 收集成功反馈
        self._collector.collect(
            feedback_type=FeedbackType.TASK_SUCCESS,
            source="lifecycle",
            content=f"任务完成: {task.description}",
            task_id=task_id,
            module_id=task.module_id,
            phase=self._lifecycle.current_phase,
            metadata={
                "estimated_minutes": task.estimated_minutes,
                "actual_minutes": task.actual_minutes,
                "efficiency": task.get_efficiency(),
            },
        )

        # 如果效率特别高，记录最佳实践
        efficiency = task.get_efficiency()
        if efficiency and efficiency > 1.5:
            self._record_best_practice(
                title=f"高效任务: {task.description}",
                content=f"任务完成效率: {efficiency:.1%}",
                task=task,
            )

    def _on_task_failed(self, event: LifecycleEvent) -> None:
        """任务失败处理."""
        task_id = event.data.get("task_id", "")
        error = event.data.get("error", "")
        task = self._tracker._tasks.get(task_id)

        if not task:
            return

        # 收集失败反馈
        self._collector.collect(
            feedback_type=FeedbackType.TASK_FAILURE,
            source="lifecycle",
            content=f"任务失败: {task.description}",
            task_id=task_id,
            module_id=task.module_id,
            phase=self._lifecycle.current_phase,
            severity="error",
            metadata={
                "error": error,
                "retry_count": task.retry_count,
            },
        )

        # 记录问题到知识库
        self._record_failure_knowledge(task, error)

    def _on_phase_change(self, event: LifecycleEvent) -> None:
        """阶段变更处理."""
        old_phase = event.data.get("old_phase")
        new_phase = event.data.get("new_phase")

        # 收集阶段反馈
        self._collector.collect(
            feedback_type=FeedbackType.SYSTEM,
            source="lifecycle",
            content=f"阶段变更: {old_phase} -> {new_phase}",
            phase=ProjectPhase(new_phase) if new_phase else None,
            metadata={
                "old_phase": old_phase,
                "new_phase": new_phase,
            },
        )

        # 阶段结束时分析和学习
        if old_phase:
            self._analyze_phase_completion(ProjectPhase(old_phase))

    def _record_best_practice(
        self,
        title: str,
        content: str,
        task: TaskProgress,
    ) -> None:
        """记录最佳实践."""
        entry = KnowledgeEntry(
            title=title,
            content=content,
            knowledge_type=KnowledgeType.BEST_PRACTICE,
            tags=[
                "high-efficiency",
                task.module_id,
                self._lifecycle.current_phase.value,
            ],
            metadata={
                "task_id": task.task_id,
                "module_id": task.module_id,
                "efficiency": task.get_efficiency(),
            },
        )
        self._kb.add_entry(entry)

    def _record_failure_knowledge(
        self,
        task: TaskProgress,
        error: str,
    ) -> None:
        """记录失败知识."""
        # 作为 bug 修复记录
        bug_entry = BugFixEntry(
            title=f"任务失败: {task.description}",
            problem=f"任务 {task.task_id} 执行失败",
            symptoms=error,
            root_cause="待分析",
            solution="待解决",
            verification="",
            tags=[
                task.module_id,
                self._lifecycle.current_phase.value,
                "task-failure",
            ],
        )
        self._kb.add_bug_fix(bug_entry)

    def _analyze_phase_completion(self, phase: ProjectPhase) -> None:
        """分析阶段完成情况."""
        # 获取该阶段的所有任务
        phase_tasks = [
            t for t in self._tracker._tasks.values()
            # 通过模块或其他方式关联阶段
        ]

        # 生成洞察
        if phase_tasks:
            insights = self._insight_gen.analyze_task_patterns(list(phase_tasks))
            for insight in insights:
                logger.info(f"发现洞察: {insight.title}")

    def collect_user_feedback(
        self,
        content: str,
        severity: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> Feedback:
        """收集用户反馈.

        Args:
            content: 反馈内容
            severity: 严重程度
            metadata: 元数据

        Returns:
            反馈条目
        """
        return self._collector.collect(
            feedback_type=FeedbackType.USER,
            source="user",
            content=content,
            severity=severity,
            phase=self._lifecycle.current_phase,
            metadata=metadata or {},
        )

    def record_bug_fix(
        self,
        problem: str,
        symptoms: str,
        root_cause: str,
        solution: str,
        verification: str = "",
        tags: list[str] | None = None,
    ) -> BugFixEntry:
        """记录 Bug 修复.

        Args:
            problem: 问题描述
            symptoms: 症状
            root_cause: 根本原因
            solution: 解决方案
            verification: 验证方法
            tags: 标签

        Returns:
            Bug 修复条目
        """
        entry = BugFixEntry(
            title=problem[:50],
            problem=problem,
            symptoms=symptoms,
            root_cause=root_cause,
            solution=solution,
            verification=verification,
            tags=tags or [],
        )

        self._kb.add_bug_fix(entry)

        # 收集反馈
        self._collector.collect(
            feedback_type=FeedbackType.BUG_FIX,
            source="user",
            content=f"Bug 修复: {problem}",
            phase=self._lifecycle.current_phase,
            metadata={
                "root_cause": root_cause,
                "solution": solution,
            },
        )

        return entry

    def get_improvement_suggestions(self) -> list[dict[str, Any]]:
        """获取改进建议.

        Returns:
            改进建议列表
        """
        suggestions = []

        # 分析任务模式
        tasks = list(self._tracker._tasks.values())
        task_insights = self._insight_gen.analyze_task_patterns(tasks)

        for insight in task_insights:
            suggestions.append({
                "type": insight.type.value,
                "title": insight.title,
                "description": insight.description,
                "recommendations": insight.recommendations,
                "confidence": insight.confidence,
            })

        # 分析未处理的反馈
        unprocessed = self._collector.get_feedbacks(unprocessed_only=True)
        if len(unprocessed) > 5:
            suggestions.append({
                "type": "warning",
                "title": "未处理反馈积压",
                "description": f"有 {len(unprocessed)} 条反馈未处理",
                "recommendations": [
                    "及时处理反馈以改进流程",
                    "考虑自动化处理常见反馈",
                ],
                "confidence": 0.9,
            })

        return suggestions

    def generate_learning_report(self) -> str:
        """生成学习报告.

        Returns:
            Markdown 格式报告
        """
        lines = [
            "# 开发学习报告",
            "",
            f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## 反馈统计",
            "",
        ]

        # 按类型统计反馈
        by_type: dict[FeedbackType, int] = {}
        for fb in self._collector._feedbacks:
            by_type[fb.type] = by_type.get(fb.type, 0) + 1

        lines.append("| 类型 | 数量 |")
        lines.append("|------|------|")
        for fb_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {fb_type.value} | {count} |")

        # 洞察列表
        if self._insight_gen._insights:
            lines.extend([
                "",
                "## 发现的洞察",
                "",
            ])

            for insight in self._insight_gen._insights:
                lines.append(f"### {insight.title}")
                lines.append("")
                lines.append(f"{insight.description}")
                lines.append("")
                lines.append(f"*置信度: {insight.confidence:.0%}*")

                if insight.recommendations:
                    lines.append("")
                    lines.append("**建议:**")
                    for rec in insight.recommendations:
                        lines.append(f"- {rec}")
                lines.append("")

        # 改进建议
        suggestions = self.get_improvement_suggestions()
        if suggestions:
            lines.extend([
                "",
                "## 改进建议",
                "",
            ])

            for sug in suggestions:
                lines.append(f"### {sug['title']}")
                lines.append("")
                lines.append(sug["description"])
                if sug["recommendations"]:
                    lines.append("")
                    for rec in sug["recommendations"]:
                        lines.append(f"- {rec}")
                lines.append("")

        return "\n".join(lines)
