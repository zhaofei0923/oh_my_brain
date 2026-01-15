"""OH MY BRAIN 测试配置."""

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建事件循环."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
