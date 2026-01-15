"""结构化日志模块测试."""

import json
import logging
import pytest

from oh_my_brain.logging import (
    LoggerAdapter,
    Span,
    StructuredFormatter,
    TraceContext,
    configure_logging,
    generate_span_id,
    generate_trace_id,
    get_logger,
    get_trace_context,
    log_function_call,
    set_trace_context,
    span_id_var,
    trace_id_var,
)


class TestTraceIdGeneration:
    """追踪 ID 生成测试."""

    def test_generate_trace_id(self):
        """测试生成追踪 ID."""
        trace_id = generate_trace_id()
        assert len(trace_id) == 16
        assert trace_id.isalnum()

    def test_generate_span_id(self):
        """测试生成 Span ID."""
        span_id = generate_span_id()
        assert len(span_id) == 8
        assert span_id.isalnum()

    def test_ids_are_unique(self):
        """测试 ID 唯一性."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestTraceContext:
    """TraceContext 测试."""

    def test_context_manager(self):
        """测试上下文管理器."""
        with TraceContext() as ctx:
            assert ctx.trace_id
            assert ctx.span_id
            assert trace_id_var.get() == ctx.trace_id
            assert span_id_var.get() == ctx.span_id

    def test_custom_trace_id(self):
        """测试自定义追踪 ID."""
        custom_id = "custom123456789"
        with TraceContext(trace_id=custom_id) as ctx:
            assert ctx.trace_id == custom_id
            assert trace_id_var.get() == custom_id

    def test_context_cleanup(self):
        """测试上下文清理."""
        original = trace_id_var.get()
        with TraceContext():
            pass
        # 上下文应该被恢复
        assert trace_id_var.get() == original


class TestSetGetTraceContext:
    """set/get_trace_context 测试."""

    def test_set_trace_context(self):
        """测试设置追踪上下文."""
        set_trace_context(
            trace_id="trace123",
            span_id="span456",
            worker_id="worker1",
            task_id="task1",
        )

        ctx = get_trace_context()
        assert ctx["trace_id"] == "trace123"
        assert ctx["span_id"] == "span456"
        assert ctx["worker_id"] == "worker1"
        assert ctx["task_id"] == "task1"


class TestSpan:
    """Span 测试."""

    def test_span_basic(self):
        """测试基本 Span."""
        with Span("test_operation") as span:
            assert span.name == "test_operation"
            assert span.span_id
            span.set_attribute("key", "value")

        assert span.duration_ms > 0
        assert span.attributes["key"] == "value"

    def test_nested_spans(self):
        """测试嵌套 Span."""
        with Span("parent") as parent:
            parent_span_id = parent.span_id
            with Span("child") as child:
                assert child.parent_span_id == parent_span_id

    def test_span_with_error(self):
        """测试带错误的 Span."""
        try:
            with Span("error_op") as span:
                raise ValueError("test error")
        except ValueError:
            pass

        assert span.attributes.get("error") is True
        assert span.attributes.get("error.type") == "ValueError"


class TestStructuredFormatter:
    """StructuredFormatter 测试."""

    def test_json_output(self):
        """测试 JSON 输出."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"
        assert "timestamp" in data
        assert "location" in data

    def test_with_trace_context(self):
        """测试带追踪上下文."""
        with TraceContext(trace_id="abc123") as ctx:
            formatter = StructuredFormatter()
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test",
                args=(),
                exc_info=None,
            )

            output = formatter.format(record)
            data = json.loads(output)

            assert "trace" in data
            assert data["trace"]["trace_id"] == "abc123"

    def test_with_exception(self):
        """测试带异常."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "traceback" in data["exception"]


class TestGetLogger:
    """get_logger 测试."""

    def test_get_logger(self):
        """测试获取日志器."""
        logger = get_logger("test.module")
        assert isinstance(logger, LoggerAdapter)

    def test_logger_with_trace(self):
        """测试日志器带追踪."""
        logger = get_logger("test")

        with TraceContext(trace_id="trace123"):
            # 日志应该自动包含追踪信息
            # 这里只验证不抛出异常
            logger.info("Test message")


class TestLogFunctionCall:
    """log_function_call 装饰器测试."""

    def test_sync_function(self):
        """测试同步函数."""
        @log_function_call()
        def my_function(a, b):
            return a + b

        result = my_function(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_async_function(self):
        """测试异步函数."""
        @log_function_call()
        async def my_async_function(a, b):
            return a + b

        result = await my_async_function(1, 2)
        assert result == 3

    def test_function_with_error(self):
        """测试带错误的函数."""
        @log_function_call()
        def error_function():
            raise ValueError("test")

        with pytest.raises(ValueError):
            error_function()


class TestConfigureLogging:
    """configure_logging 测试."""

    def test_configure_json(self):
        """测试配置 JSON 格式."""
        configure_logging(format_type="json")
        # 验证不抛出异常

    def test_configure_human(self):
        """测试配置人类可读格式."""
        configure_logging(format_type="human")
        # 验证不抛出异常
