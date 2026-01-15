"""Prometheus 指标导出.

提供系统运行时指标，用于监控和告警。
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """指标值."""
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Counter:
    """计数器指标."""

    def __init__(self, name: str, description: str, labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = defaultdict(float)

    def inc(self, value: float = 1.0, **labels) -> None:
        """增加计数."""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        self._values[label_key] += value

    def get(self, **labels) -> float:
        """获取当前值."""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        return self._values[label_key]

    def collect(self) -> list[MetricValue]:
        """收集所有值."""
        result = []
        for label_key, value in self._values.items():
            labels = dict(zip(self.label_names, label_key))
            result.append(MetricValue(value=value, labels=labels))
        return result


class Gauge:
    """仪表盘指标."""

    def __init__(self, name: str, description: str, labels: list[str] | None = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = {}

    def set(self, value: float, **labels) -> None:
        """设置值."""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        self._values[label_key] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """增加值."""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        self._values[label_key] = self._values.get(label_key, 0) + value

    def dec(self, value: float = 1.0, **labels) -> None:
        """减少值."""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        self._values[label_key] = self._values.get(label_key, 0) - value

    def get(self, **labels) -> float:
        """获取当前值."""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        return self._values.get(label_key, 0)

    def collect(self) -> list[MetricValue]:
        """收集所有值."""
        result = []
        for label_key, value in self._values.items():
            labels = dict(zip(self.label_names, label_key))
            result.append(MetricValue(value=value, labels=labels))
        return result


class Histogram:
    """直方图指标."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[tuple, dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: dict[tuple, float] = defaultdict(float)
        self._totals: dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, **labels) -> None:
        """观察一个值."""
        label_key = tuple(labels.get(l, "") for l in self.label_names)
        self._sums[label_key] += value
        self._totals[label_key] += 1
        for bucket in self.buckets:
            if value <= bucket:
                self._counts[label_key][bucket] += 1

    def time(self, **labels):
        """计时上下文管理器."""
        return HistogramTimer(self, labels)

    def collect(self) -> list[MetricValue]:
        """收集所有值."""
        result = []
        for label_key in set(self._counts.keys()) | set(self._sums.keys()):
            labels = dict(zip(self.label_names, label_key))

            # bucket 值
            cumulative = 0
            for bucket in self.buckets:
                cumulative += self._counts[label_key].get(bucket, 0)
                bucket_labels = {**labels, "le": str(bucket)}
                result.append(MetricValue(value=cumulative, labels=bucket_labels))

            # +Inf bucket
            inf_labels = {**labels, "le": "+Inf"}
            result.append(MetricValue(value=self._totals[label_key], labels=inf_labels))

            # sum 和 count
            result.append(MetricValue(
                value=self._sums[label_key],
                labels={**labels, "_type": "sum"},
            ))
            result.append(MetricValue(
                value=self._totals[label_key],
                labels={**labels, "_type": "count"},
            ))

        return result


class HistogramTimer:
    """直方图计时器."""

    def __init__(self, histogram: Histogram, labels: dict[str, str]):
        self._histogram = histogram
        self._labels = labels
        self._start: float | None = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._start is not None:
            duration = time.perf_counter() - self._start
            self._histogram.observe(duration, **self._labels)


class MetricsRegistry:
    """指标注册表."""

    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}

    def counter(self, name: str, description: str, labels: list[str] | None = None) -> Counter:
        """创建或获取计数器."""
        if name not in self._metrics:
            self._metrics[name] = Counter(name, description, labels)
        return self._metrics[name]

    def gauge(self, name: str, description: str, labels: list[str] | None = None) -> Gauge:
        """创建或获取仪表盘."""
        if name not in self._metrics:
            self._metrics[name] = Gauge(name, description, labels)
        return self._metrics[name]

    def histogram(
        self,
        name: str,
        description: str,
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """创建或获取直方图."""
        if name not in self._metrics:
            self._metrics[name] = Histogram(name, description, labels, buckets)
        return self._metrics[name]

    def collect_all(self) -> dict[str, list[MetricValue]]:
        """收集所有指标."""
        result = {}
        for name, metric in self._metrics.items():
            result[name] = metric.collect()
        return result

    def export_prometheus(self) -> str:
        """导出 Prometheus 格式."""
        lines = []
        for name, metric in self._metrics.items():
            # HELP
            lines.append(f"# HELP {name} {metric.description}")

            # TYPE
            if isinstance(metric, Counter):
                lines.append(f"# TYPE {name} counter")
            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {name} gauge")
            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {name} histogram")

            # 值
            for mv in metric.collect():
                if mv.labels:
                    # 过滤内部标签
                    display_labels = {k: v for k, v in mv.labels.items() if not k.startswith("_")}
                    if display_labels:
                        label_str = ",".join(f'{k}="{v}"' for k, v in display_labels.items())
                        if isinstance(metric, Histogram) and "le" in mv.labels:
                            lines.append(f"{name}_bucket{{{label_str}}} {mv.value}")
                        elif mv.labels.get("_type") == "sum":
                            lines.append(f"{name}_sum{{{label_str.replace(',le=\"+Inf\"', '')}}} {mv.value}")
                        elif mv.labels.get("_type") == "count":
                            lines.append(f"{name}_count{{{label_str.replace(',le=\"+Inf\"', '')}}} {mv.value}")
                        else:
                            lines.append(f"{name}{{{label_str}}} {mv.value}")
                    else:
                        lines.append(f"{name} {mv.value}")
                else:
                    lines.append(f"{name} {mv.value}")

        return "\n".join(lines)


# 全局注册表
REGISTRY = MetricsRegistry()


# ============================================================
# OH MY BRAIN 特定指标
# ============================================================

# Brain 指标
brain_tasks_total = REGISTRY.counter(
    "omb_brain_tasks_total",
    "Total number of tasks processed",
    ["status", "task_type"],
)

brain_workers_active = REGISTRY.gauge(
    "omb_brain_workers_active",
    "Number of active workers",
)

brain_tasks_pending = REGISTRY.gauge(
    "omb_brain_tasks_pending",
    "Number of pending tasks",
)

brain_tasks_running = REGISTRY.gauge(
    "omb_brain_tasks_running",
    "Number of running tasks",
)

brain_task_duration_seconds = REGISTRY.histogram(
    "omb_brain_task_duration_seconds",
    "Task execution duration in seconds",
    ["task_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

# Worker 指标
worker_requests_total = REGISTRY.counter(
    "omb_worker_requests_total",
    "Total number of requests to Brain",
    ["request_type", "status"],
)

worker_llm_calls_total = REGISTRY.counter(
    "omb_worker_llm_calls_total",
    "Total number of LLM API calls",
    ["model", "status"],
)

worker_llm_tokens_total = REGISTRY.counter(
    "omb_worker_llm_tokens_total",
    "Total number of tokens used",
    ["model", "type"],  # type: input/output
)

worker_llm_latency_seconds = REGISTRY.histogram(
    "omb_worker_llm_latency_seconds",
    "LLM API call latency in seconds",
    ["model"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

# 安全检查指标
safety_checks_total = REGISTRY.counter(
    "omb_safety_checks_total",
    "Total number of safety checks",
    ["command_type", "result"],  # result: approved/rejected
)

# 连接指标
connections_total = REGISTRY.counter(
    "omb_connections_total",
    "Total number of connection events",
    ["component", "event"],  # event: connect/disconnect/reconnect
)


def get_metrics_endpoint():
    """获取 Prometheus 指标端点的响应."""
    return REGISTRY.export_prometheus()
