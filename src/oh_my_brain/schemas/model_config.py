"""AI模型配置数据模型.

支持用户自定义多个AI模型，按任务类型选择不同模型。
"""

from enum import Enum

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """模型提供商（API协议类型）."""

    ANTHROPIC = "anthropic"  # Anthropic API格式（MiniMax兼容）
    OPENAI = "openai"  # OpenAI API格式（DeepSeek, Claude兼容）


class ModelCapability(str, Enum):
    """模型能力标签."""

    CODE = "code"  # 代码生成
    REASONING = "reasoning"  # 推理
    PLANNING = "planning"  # 规划
    REVIEW = "review"  # 代码审查
    DOCS = "docs"  # 文档生成


class ModelConfig(BaseModel):
    """单个AI模型配置.

    用户可配置多个模型，按需选择。
    """

    name: str = Field(..., description="模型名称（用户自定义，用于引用）")
    provider: ModelProvider = Field(..., description="API协议类型")
    api_base: str = Field(..., description="API基础URL")
    model: str = Field(..., description="模型标识符")
    api_key_env: str = Field(
        ..., description="API Key环境变量名（不直接存储密钥）"
    )

    # 成本与性能参考（帮助用户决策）
    cost_per_1k_tokens: float = Field(
        0.0, ge=0, description="每1000 token费用（美元），用于成本预估"
    )
    avg_latency_ms: int = Field(
        1000, ge=0, description="平均响应延迟（毫秒），用于性能参考"
    )

    # 能力标签
    capabilities: list[ModelCapability] = Field(
        default_factory=list, description="模型能力标签，用于Brain建议"
    )

    # 高级配置
    max_tokens: int = Field(4096, description="最大输出token数")
    temperature: float = Field(0.7, ge=0, le=2, description="温度参数")
    timeout_seconds: int = Field(120, description="请求超时时间")

    # 健康状态（运行时更新）
    is_healthy: bool = Field(True, description="模型是否健康可用")
    consecutive_failures: int = Field(0, description="连续失败次数")

    model_config = {"extra": "forbid"}

    def get_api_key_env_name(self) -> str:
        """获取API Key环境变量名."""
        return self.api_key_env


class TaskModelMapping(BaseModel):
    """任务类型到模型的映射.

    用户可自定义不同任务使用哪个模型。
    """

    planning: str | None = Field(None, description="规划任务使用的模型")
    coding: str | None = Field(None, description="编码任务使用的模型")
    review: str | None = Field(None, description="审查任务使用的模型")
    testing: str | None = Field(None, description="测试任务使用的模型")
    docs: str | None = Field(None, description="文档任务使用的模型")
    refactor: str | None = Field(None, description="重构任务使用的模型")
    default: str = Field(..., description="默认模型（必填）")

    model_config = {"extra": "allow"}  # 允许用户自定义任务类型

    def get_model_for_task(self, task_type: str) -> str:
        """根据任务类型获取模型名称.

        Args:
            task_type: 任务类型

        Returns:
            模型名称
        """
        mapping = self.model_dump()
        return mapping.get(task_type) or mapping["default"]


class BrainSuggestionConfig(BaseModel):
    """Brain建议配置."""

    enabled: bool = Field(True, description="是否启用Brain建议")
    auto_apply: bool = Field(
        False, description="是否自动应用建议（False则需用户确认）"
    )
    consider_cost: bool = Field(True, description="建议时考虑成本因素")
    consider_latency: bool = Field(True, description="建议时考虑延迟因素")
    consider_capability: bool = Field(True, description="建议时考虑能力匹配")


class ModelPoolConfig(BaseModel):
    """模型池完整配置.

    对应用户的 models.yaml 配置文件。
    """

    models: list[ModelConfig] = Field(..., min_length=1, description="可用模型列表")
    task_model_mapping: TaskModelMapping = Field(..., description="任务-模型映射")
    brain_suggestions: BrainSuggestionConfig = Field(
        default_factory=BrainSuggestionConfig, description="Brain建议配置"
    )

    # 高可用配置
    health_check_interval_seconds: int = Field(
        30, description="健康检查间隔（秒）"
    )
    max_consecutive_failures: int = Field(
        3, description="最大连续失败次数，超过则标记不健康"
    )
    circuit_breaker_reset_seconds: int = Field(
        60, description="熔断器重置时间（秒）"
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "models": [
                        {
                            "name": "deepseek-coder",
                            "provider": "openai",
                            "api_base": "https://api.deepseek.com",
                            "model": "deepseek-coder",
                            "api_key_env": "DEEPSEEK_API_KEY",
                            "cost_per_1k_tokens": 0.001,
                            "capabilities": ["code"],
                        },
                        {
                            "name": "minimax-m21",
                            "provider": "anthropic",
                            "api_base": "https://api.minimax.io",
                            "model": "MiniMax-M2.1",
                            "api_key_env": "MINIMAX_API_KEY",
                            "cost_per_1k_tokens": 0.002,
                            "capabilities": ["code", "reasoning", "planning"],
                        },
                    ],
                    "task_model_mapping": {
                        "planning": "minimax-m21",
                        "coding": "deepseek-coder",
                        "default": "deepseek-coder",
                    },
                }
            ]
        },
    }

    def get_model_by_name(self, name: str) -> ModelConfig | None:
        """根据名称获取模型配置."""
        for model in self.models:
            if model.name == name:
                return model
        return None

    def get_healthy_models(self) -> list[ModelConfig]:
        """获取所有健康的模型."""
        return [m for m in self.models if m.is_healthy]

    def get_models_with_capability(
        self, capability: ModelCapability
    ) -> list[ModelConfig]:
        """获取具有指定能力的模型."""
        return [m for m in self.models if capability in m.capabilities and m.is_healthy]
