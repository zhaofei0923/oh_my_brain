"""LLM客户端.

统一的LLM调用接口，支持多种模型提供商。
"""

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM客户端基类."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """生成补全.

        Args:
            prompt: 提示
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        pass

    @abstractmethod
    async def stream_complete(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成补全.

        Args:
            prompt: 提示
            **kwargs: 其他参数

        Yields:
            生成的文本片段
        """
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI/兼容API客户端."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None

    async def _get_client(self):
        """获取客户端."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        """生成补全."""
        client = await self._get_client()

        response = await client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        return response.choices[0].message.content or ""

    async def stream_complete(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成补全."""
        client = await self._get_client()

        stream = await client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude客户端."""

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    async def _get_client(self):
        """获取客户端."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        """生成补全."""
        client = await self._get_client()

        response = await client.messages.create(
            model=self._model_name,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    async def stream_complete(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成补全."""
        client = await self._get_client()

        async with client.messages.stream(
            model=self._model_name,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class LLMClient:
    """统一LLM客户端.

    根据模型名称自动选择合适的客户端。
    """

    # 模型前缀到客户端的映射
    MODEL_PREFIXES = {
        "gpt-": "openai",
        "claude-": "anthropic",
        "deepseek-": "openai",  # DeepSeek使用OpenAI兼容API
        "minimax-": "openai",  # MiniMax使用OpenAI兼容API
    }

    # 特殊模型的API配置
    MODEL_CONFIGS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "env_key": "DEEPSEEK_API_KEY",
        },
        "minimax": {
            "base_url": "https://api.minimax.chat/v1",
            "env_key": "MINIMAX_API_KEY",
        },
    }

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._model_name = model_name
        self._client = self._create_client(model_name, api_key, base_url)

    def _create_client(
        self,
        model_name: str,
        api_key: str | None,
        base_url: str | None,
    ) -> BaseLLMClient:
        """创建客户端."""
        # 确定客户端类型
        client_type = "openai"  # 默认
        for prefix, ctype in self.MODEL_PREFIXES.items():
            if model_name.startswith(prefix):
                client_type = ctype
                break

        # 检查特殊配置
        for provider, config in self.MODEL_CONFIGS.items():
            if model_name.startswith(provider):
                base_url = base_url or config["base_url"]
                api_key = api_key or os.getenv(config["env_key"])
                break

        # 创建客户端
        if client_type == "anthropic":
            return AnthropicClient(model_name=model_name, api_key=api_key)
        else:
            return OpenAIClient(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
            )

    async def complete(self, prompt: str, **kwargs) -> str:
        """生成补全."""
        return await self._client.complete(prompt, **kwargs)

    async def stream_complete(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成补全."""
        async for text in self._client.stream_complete(prompt, **kwargs):
            yield text
