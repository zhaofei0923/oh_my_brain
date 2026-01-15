"""Prometheus 指标模块测试."""

import pytest

from oh_my_brain.brain.metrics import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    brain_tasks_total,
    brain_workers_active,
    get_metrics_endpoint,
)


class TestCounter:
    """Counter 测试."""

    def test_basic_increment(self):
        """测试基本增加."""
        counter = Counter("test_counter", "Test counter")
        counter.inc()
        assert counter.get() == 1.0
        counter.inc(5)
        assert counter.get() == 6.0

    def test_labeled_counter(self):
        """测试带标签的计数器."""
        counter = Counter("test_labeled", "Test", ["status", "type"])
        counter.inc(status="success", type="task")
        counter.inc(status="error", type="task")
        counter.inc(2, status="success", type="task")

        assert counter.get(status="success", type="task") == 3.0
        assert counter.get(status="error", type="task") == 1.0
        assert counter.get(status="pending", type="task") == 0.0

    def test_collect(self):
        """测试收集指标."""
        counter = Counter("test_collect", "Test", ["label"])
        counter.inc(label="a")
        counter.inc(2, label="b")

        values = counter.collect()
        assert len(values) == 2


class TestGauge:
    """Gauge 测试."""

    def test_set_and_get(self):
        """测试设置和获取."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(42)
        assert gauge.get() == 42

    def test_inc_dec(self):
        """测试增减."""
        gauge = Gauge("test_gauge2", "Test gauge")
        gauge.inc()
        assert gauge.get() == 1
        gauge.inc(5)
        assert gauge.get() == 6
        gauge.dec(2)
        assert gauge.get() == 4

    def test_labeled_gauge(self):
        """测试带标签的仪表."""
        gauge = Gauge("test_labeled_gauge", "Test", ["worker"])
        gauge.set(10, worker="w1")
        gauge.set(20, worker="w2")

        assert gauge.get(worker="w1") == 10
        assert gauge.get(worker="w2") == 20


class TestHistogram:
    """Histogram 测试."""

    def test_observe(self):
        """测试观察值."""
        hist = Histogram("test_hist", "Test histogram", buckets=(1, 5, 10))
        hist.observe(0.5)
        hist.observe(3)
        hist.observe(7)
        hist.observe(15)

        values = hist.collect()
        # 应该有 bucket 值 + sum + count
        assert len(values) > 0

    def test_timer(self):
        """测试计时器."""
        import time

        hist = Histogram("test_timer", "Test timer")
        with hist.time():
            time.sleep(0.01)

        values = hist.collect()
        assert len(values) > 0


class TestMetricsRegistry:
    """MetricsRegistry 测试."""

    def test_create_metrics(self):
        """测试创建指标."""
        registry = MetricsRegistry()

        counter = registry.counter("my_counter", "A counter")
        gauge = registry.gauge("my_gauge", "A gauge")
        hist = registry.histogram("my_hist", "A histogram")

        assert isinstance(counter, Counter)
        assert isinstance(gauge, Gauge)
        assert isinstance(hist, Histogram)

    def test_get_existing_metric(self):
        """测试获取已存在的指标."""
        registry = MetricsRegistry()

        counter1 = registry.counter("same_name", "Counter")
        counter2 = registry.counter("same_name", "Counter")

        assert counter1 is counter2

    def test_export_prometheus(self):
        """测试导出 Prometheus 格式."""
        registry = MetricsRegistry()

        counter = registry.counter("export_test", "Test export", ["status"])
        counter.inc(status="ok")

        output = registry.export_prometheus()
        assert "export_test" in output
        assert "# HELP" in output
        assert "# TYPE" in output


class TestGlobalMetrics:
    """全局指标测试."""

    def test_brain_metrics_exist(self):
        """测试 Brain 指标存在."""
        assert brain_tasks_total is not None
        assert brain_workers_active is not None

    def test_get_metrics_endpoint(self):
        """测试获取指标端点."""
        output = get_metrics_endpoint()
        assert isinstance(output, str)
        # 应该包含至少一些定义的指标
        assert "omb_" in output or output == ""  # 可能还没有任何值


class TestIntegration:
    """集成测试."""

    def test_full_workflow(self):
        """测试完整工作流."""
        # 模拟任务处理流程
        brain_workers_active.inc()
        brain_tasks_total.inc(status="completed", task_type="dev")

        output = get_metrics_endpoint()
        assert "omb_brain_workers_active" in output
        assert "omb_brain_tasks_total" in output
