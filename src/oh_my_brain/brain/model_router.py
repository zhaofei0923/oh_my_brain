"""模型路由器.

根据任务类型选择合适的AI模型，支持用户自定义配置。
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from oh_my_brain.schemas.model_config import (
    ModelCapability,
    ModelConfig,
    ModelPoolConfig,
    TaskModelMapping,
)
from oh_my_brain.schemas.task import TaskType

logger = logging.getLogger(__name__)


class ModelSuggestion:
    """模型建议."""

    def __init__(
        self,
        model_name: str,
        reason: str,
        estimated_cost: float = 0.0,
        estimated_latency_ms: int = 0,
    ):
        self.model_name = model_name
        self.reason = reason
        self.estimated_cost = estimated_cost
        self.estimated_latency_ms = estimated_latency_ms


class ModelRouter:
    """模型路由器.

    功能：
    1. 根据任务类型选择模型（用户配置优先）
    2. 提供模型建议（考虑费用、延迟、能力）
    3. 模型健康检查和故障转移
    """

    def __init__(self, config: ModelPoolConfig):
        self._config = config
        self._models: dict[str, ModelConfig] = {m.name: m for m in config.models}

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ModelRouter":
        """从YAML文件加载配置."""
        path = Path(path).expanduser()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = ModelPoolConfig.model_validate(data)
        return cls(config)

    def get_model(self, name: str) -> ModelConfig | None:
        """根据名称获取模型配置."""
        return self._models.get(name)

    def get_model_for_task(
        self,
        task_type: TaskType,
        preferred_model: str | None = None,
    ) -> ModelConfig | None:
        """根据任务类型获取模型.

        Args:
            task_type: 任务类型
            preferred_model: 用户指定的模型（优先级最高）

        Returns:
            模型配置
        """
        # 用户指定优先
        if preferred_model:
            model = self._models.get(preferred_model)
            if model and model.is_healthy:
                return model
            logger.warning(f"Preferred model {preferred_model} not available")

        # 按任务类型映射
        model_name = self._config.task_model_mapping.get_model_for_task(task_type.value)
        model = self._models.get(model_name)

        if model and model.is_healthy:
            return model

        # 回退到默认模型
        default_name = self._config.task_model_mapping.default
        return self._models.get(default_name)

    def suggest_model(
        self,
        task_type: TaskType,
        estimated_tokens: int = 1000,
        prefer_low_cost: bool = True,
        prefer_low_latency: bool = False,
    ) -> ModelSuggestion | None:
        """为任务提供模型建议.

        Brain根据多种因素给出建议，但最终由用户决定。

        Args:
            task_type: 任务类型
            estimated_tokens: 预估token数量
            prefer_low_cost: 是否优先考虑低成本
            prefer_low_latency: 是否优先考虑低延迟

        Returns:
            模型建议
        """
        if not self._config.brain_suggestions.enabled:
            return None

        # 获取具有相关能力的健康模型
        capability = self._task_type_to_capability(task_type)
        candidates = self._config.get_models_with_capability(capability)

        if not candidates:
            candidates = self._config.get_healthy_models()

        if not candidates:
            return None

        # 评分
        scored_models = []
        for model in candidates:
            score = self._score_model(
                model,
                estimated_tokens,
                prefer_low_cost,
                prefer_low_latency,
            )
            scored_models.append((score, model))

        scored_models.sort(key=lambda x: x[0], reverse=True)
        best_model = scored_models[0][1]

        # 构建建议理由
        reasons = []
        if capability in best_model.capabilities:
            reasons.append(f"具备{capability.value}能力")
        if prefer_low_cost:
            reasons.append(f"成本${best_model.cost_per_1k_tokens}/1k tokens")
        if prefer_low_latency:
            reasons.append(f"延迟{best_model.avg_latency_ms}ms")

        return ModelSuggestion(
            model_name=best_model.name,
            reason="、".join(reasons) if reasons else "默认推荐",
            estimated_cost=(estimated_tokens / 1000) * best_model.cost_per_1k_tokens,
            estimated_latency_ms=best_model.avg_latency_ms,
        )

    def _task_type_to_capability(self, task_type: TaskType) -> ModelCapability:
        """任务类型映射到模型能力."""
        mapping = {
            TaskType.PLANNING: ModelCapability.PLANNING,
            TaskType.CODING: ModelCapability.CODE,
            TaskType.REVIEW: ModelCapability.REVIEW,
            TaskType.TESTING: ModelCapability.CODE,
            TaskType.DOCS: ModelCapability.DOCS,
            TaskType.REFACTOR: ModelCapability.CODE,
        }
        return mapping.get(task_type, ModelCapability.CODE)

    def _score_model(
        self,
        model: ModelConfig,
        estimated_tokens: int,
        prefer_low_cost: bool,
        prefer_low_latency: bool,
    ) -> float:
        """评分模型.

        Returns:
            评分（越高越好）
        """
        score = 0.0

        # 基础分
        score += 50

        # 成本因素
        if prefer_low_cost:
            # 成本越低分越高
            max_cost = max(m.cost_per_1k_tokens for m in self._config.models) or 1
            cost_score = (1 - model.cost_per_1k_tokens / max_cost) * 30
            score += cost_score

        # 延迟因素
        if prefer_low_latency:
            max_latency = max(m.avg_latency_ms for m in self._config.models) or 1
            latency_score = (1 - model.avg_latency_ms / max_latency) * 20
            score += latency_score

        # 健康度惩罚
        if model.consecutive_failures > 0:
            score -= model.consecutive_failures * 10

        return score

    def mark_model_failure(self, model_name: str) -> None:
        """标记模型失败."""
        model = self._models.get(model_name)
        if model:
            model.consecutive_failures += 1
            if model.consecutive_failures >= self._config.max_consecutive_failures:
                model.is_healthy = False
                logger.warning(f"Model {model_name} marked as unhealthy")

    def mark_model_success(self, model_name: str) -> None:
        """标记模型成功."""
        model = self._models.get(model_name)
        if model:
            model.consecutive_failures = 0
            model.is_healthy = True

    def reset_model_health(self, model_name: str) -> None:
        """重置模型健康状态（熔断器重置）."""
        model = self._models.get(model_name)
        if model:
            model.is_healthy = True
            model.consecutive_failures = 0

    def get_model_api_key(self, model_name: str) -> str | None:
        """获取模型API Key（从环境变量）."""
        model = self._models.get(model_name)
        if model:
            return os.environ.get(model.api_key_env)
        return None

    def get_status(self) -> dict[str, Any]:
        """获取模型路由器状态."""
        return {
            "total_models": len(self._models),
            "healthy_models": len(self._config.get_healthy_models()),
            "models": {
                name: {
                    "healthy": m.is_healthy,
                    "failures": m.consecutive_failures,
                    "provider": m.provider.value,
                }
                for name, m in self._models.items()
            },
        }
