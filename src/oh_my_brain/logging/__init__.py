"""结构化日志模块.

提供 JSON 格式日志输出和分布式追踪支持。
"""

import contextvars
import json
import logging
import sys
import time
import traceback
import uuid
from datetime import datetime
from typing import Any

# 上下文变量用于存储追踪信息
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
span_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("span_id", default="")
worker_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("worker_id", default="")
task_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="")


def generate_trace_id() -> str:
    """生成追踪 ID."""
    return uuid.uuid4().hex[:16]


def generate_span_id() -> str:
    """生成 Span ID."""
    return uuid.uuid4().hex[:8]


def set_trace_context(
    trace_id: str | None = None,
    span_id: str | None = None,
    worker_id: str | None = None,
    task_id: str | None = None,
) -> None:
    """设置追踪上下文."""
    if trace_id:
        trace_id_var.set(trace_id)
    if span_id:
        span_id_var.set(span_id)
    if worker_id:
        worker_id_var.set(worker_id)
    if task_id:
        task_id_var.set(task_id)


def get_trace_context() -> dict[str, str]:
    """获取当前追踪上下文."""
    return {
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "worker_id": worker_id_var.get(),
        "task_id": task_id_var.get(),
    }


class TraceContext:
    """追踪上下文管理器."""

    def __init__(
        self,
        trace_id: str | None = None,
        worker_id: str | None = None,
        task_id: str | None = None,
    ):
        self._trace_id = trace_id or generate_trace_id()
        self._span_id = generate_span_id()
        self._worker_id = worker_id or ""
        self._task_id = task_id or ""
        self._tokens: list[contextvars.Token] = []

    def __enter__(self) -> "TraceContext":
        self._tokens.append(trace_id_var.set(self._trace_id))
        self._tokens.append(span_id_var.set(self._span_id))
        if self._worker_id:
            self._tokens.append(worker_id_var.set(self._worker_id))
        if self._task_id:
            self._tokens.append(task_id_var.set(self._task_id))
        return self

    def __exit__(self, *args) -> None:
        for token in reversed(self._tokens):
            try:
                token.var.reset(token)
            except ValueError:
                pass

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_id


class Span:
    """追踪 Span.

    用于记录单个操作的开始和结束时间。
    """

    def __init__(self, name: str, parent_span_id: str | None = None):
        self.name = name
        self.span_id = generate_span_id()
        self.parent_span_id = parent_span_id or span_id_var.get()
        self.start_time: float = 0
        self.end_time: float = 0
        self.attributes: dict[str, Any] = {}
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "Span":
        self.start_time = time.time()
        self._token = span_id_var.set(self.span_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()
        if self._token:
            span_id_var.reset(self._token)

        # 记录 span 完成日志
        duration_ms = (self.end_time - self.start_time) * 1000
        logger = logging.getLogger(__name__)

        if exc_type:
            self.attributes["error"] = True
            self.attributes["error.type"] = exc_type.__name__
            self.attributes["error.message"] = str(exc_val)
            logger.warning(
                f"Span '{self.name}' completed with error",
                extra={
                    "span_name": self.name,
                    "duration_ms": duration_ms,
                    "parent_span_id": self.parent_span_id,
                    **self.attributes,
                },
            )
        else:
            logger.debug(
                f"Span '{self.name}' completed",
                extra={
                    "span_name": self.name,
                    "duration_ms": duration_ms,
                    "parent_span_id": self.parent_span_id,
                    **self.attributes,
                },
            )

    def set_attribute(self, key: str, value: Any) -> None:
        """设置属性."""
        self.attributes[key] = value

    @property
    def duration_ms(self) -> float:
        """获取持续时间（毫秒）."""
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0


class StructuredFormatter(logging.Formatter):
    """结构化 JSON 日志格式化器."""

    def __init__(
        self,
        include_trace: bool = True,
        include_extra: bool = True,
        timestamp_format: str = "iso",
    ):
        super().__init__()
        self._include_trace = include_trace
        self._include_extra = include_extra
        self._timestamp_format = timestamp_format
        self._skip_keys = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
        }

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON."""
        # 基础字段
        log_dict: dict[str, Any] = {
            "timestamp": self._format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 位置信息
        log_dict["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # 追踪信息
        if self._include_trace:
            trace_context = get_trace_context()
            if any(trace_context.values()):
                log_dict["trace"] = {
                    k: v for k, v in trace_context.items() if v
                }

        # 额外字段
        if self._include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in self._skip_keys:
                    try:
                        # 确保可以序列化
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)
            if extra:
                log_dict["extra"] = extra

        # 异常信息
        if record.exc_info:
            log_dict["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_dict, ensure_ascii=False, default=str)

    def _format_timestamp(self, timestamp: float) -> str:
        """格式化时间戳."""
        if self._timestamp_format == "iso":
            return datetime.fromtimestamp(timestamp).isoformat()
        elif self._timestamp_format == "unix":
            return str(timestamp)
        else:
            return datetime.fromtimestamp(timestamp).strftime(self._timestamp_format)


class HumanReadableFormatter(logging.Formatter):
    """人类可读的日志格式化器（带颜色）."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self._use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录."""
        # 时间戳
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # 级别（带颜色）
        if self._use_colors:
            color = self.COLORS.get(record.levelname, "")
            level = f"{color}{record.levelname:8}{self.RESET}"
        else:
            level = f"{record.levelname:8}"

        # 追踪信息
        trace_id = trace_id_var.get()
        trace_str = f"[{trace_id[:8]}] " if trace_id else ""

        # 位置
        location = f"{record.name}:{record.lineno}"

        # 消息
        message = record.getMessage()

        # 组装
        line = f"{timestamp} {level} {trace_str}{location} - {message}"

        # 异常
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def configure_logging(
    level: int | str = logging.INFO,
    format_type: str = "human",  # "human" 或 "json"
    include_trace: bool = True,
    stream: Any = None,
) -> None:
    """配置日志系统.

    Args:
        level: 日志级别
        format_type: 格式类型 ("human" 或 "json")
        include_trace: 是否包含追踪信息
        stream: 输出流（默认 stderr）
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建处理器
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    # 设置格式化器
    if format_type == "json":
        handler.setFormatter(StructuredFormatter(include_trace=include_trace))
    else:
        handler.setFormatter(HumanReadableFormatter())

    root_logger.addHandler(handler)


class LoggerAdapter(logging.LoggerAdapter):
    """带追踪上下文的日志适配器."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None):
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """处理日志消息，添加追踪上下文."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        extra.update(get_trace_context())
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> LoggerAdapter:
    """获取带追踪支持的日志器.

    Args:
        name: 日志器名称

    Returns:
        LoggerAdapter 实例
    """
    return LoggerAdapter(logging.getLogger(name))


# ============================================================
# 日志装饰器
# ============================================================


def log_function_call(logger: logging.Logger | None = None):
    """装饰器：记录函数调用."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with Span(func.__name__) as span:
                    span.set_attribute("args_count", len(args))
                    span.set_attribute("kwargs_keys", list(kwargs.keys()))
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with Span(func.__name__) as span:
                    span.set_attribute("args_count", len(args))
                    span.set_attribute("kwargs_keys", list(kwargs.keys()))
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        raise
            return sync_wrapper

    return decorator


# 需要导入 asyncio
import asyncio
