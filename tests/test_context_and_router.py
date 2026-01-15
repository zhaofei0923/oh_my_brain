"""Context Manager 和 Model Router 测试."""

import pytest
from unittest.mock import AsyncMock, patch

from oh_my_brain.brain.context_manager import ContextManager
from oh_my_brain.brain.model_router import ModelRouter


class TestContextManager:
    """测试上下文管理器."""

    @pytest.fixture
    def context_manager(self):
        """创建上下文管理器（内存模式）."""
        return ContextManager(redis_url=None)

    @pytest.mark.asyncio
    async def test_add_and_get_message(self, context_manager):
        """测试添加和获取消息."""
        task_id = "task-001"

        await context_manager.add_message(
            task_id=task_id,
            role="user",
            content="Hello",
        )

        messages = await context_manager.get_messages(task_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_empty_context(self, context_manager):
        """测试获取空上下文."""
        messages = await context_manager.get_messages("nonexistent")
        assert messages == []

    @pytest.mark.asyncio
    async def test_add_note(self, context_manager):
        """测试添加笔记."""
        task_id = "task-001"

        await context_manager.add_note(
            task_id=task_id,
            category="discovery",
            content="Found an issue",
        )

        notes = await context_manager.get_notes(task_id)
        assert len(notes) == 1
        assert notes[0]["category"] == "discovery"

    @pytest.mark.asyncio
    async def test_clear_context(self, context_manager):
        """测试清除上下文."""
        task_id = "task-001"

        await context_manager.add_message(task_id, "user", "Hello")
        await context_manager.clear_context(task_id)

        messages = await context_manager.get_messages(task_id)
        assert messages == []

    @pytest.mark.asyncio
    async def test_count_tokens(self, context_manager):
        """测试token计数."""
        task_id = "task-001"

        await context_manager.add_message(task_id, "user", "Hello world")
        count = await context_manager.count_tokens(task_id)

        assert count > 0

    @pytest.mark.asyncio
    async def test_multiple_messages(self, context_manager):
        """测试多条消息."""
        task_id = "task-001"

        await context_manager.add_message(task_id, "user", "Hello")
        await context_manager.add_message(task_id, "assistant", "Hi there!")
        await context_manager.add_message(task_id, "user", "How are you?")

        messages = await context_manager.get_messages(task_id)
        assert len(messages) == 3


class TestModelRouter:
    """测试模型路由器."""

    @pytest.fixture
    def sample_config(self, tmp_path):
        """创建示例配置文件."""
        config_content = """
models:
  - name: "gpt-4"
    provider: "openai"
    model: "gpt-4"
    api_key_env: "OPENAI_API_KEY"
    capabilities: [code, reasoning]
    cost_per_1k_tokens: 0.03

  - name: "deepseek-coder"
    provider: "openai"
    api_base: "https://api.deepseek.com"
    model: "deepseek-coder"
    api_key_env: "DEEPSEEK_API_KEY"
    capabilities: [code]
    cost_per_1k_tokens: 0.001

task_model_mapping:
  planning: "gpt-4"
  coding: "deepseek-coder"
  default: "deepseek-coder"
"""
        config_file = tmp_path / "models.yaml"
        config_file.write_text(config_content)
        return config_file

    def test_load_from_yaml(self, sample_config):
        """测试从YAML加载配置."""
        router = ModelRouter(config_path=sample_config)

        assert len(router._models) == 2
        assert "gpt-4" in router._models
        assert "deepseek-coder" in router._models

    def test_get_model_for_task_type(self, sample_config):
        """测试根据任务类型获取模型."""
        router = ModelRouter(config_path=sample_config)

        planning_model = router.get_model_for_task("planning")
        assert planning_model is not None
        assert planning_model.name == "gpt-4"

        coding_model = router.get_model_for_task("coding")
        assert coding_model is not None
        assert coding_model.name == "deepseek-coder"

    def test_default_model(self, sample_config):
        """测试默认模型."""
        router = ModelRouter(config_path=sample_config)

        unknown_model = router.get_model_for_task("unknown_task")
        assert unknown_model is not None
        assert unknown_model.name == "deepseek-coder"

    def test_user_specified_model(self, sample_config):
        """测试用户指定模型."""
        router = ModelRouter(config_path=sample_config)

        model = router.get_model_for_task("coding", user_model="gpt-4")
        assert model.name == "gpt-4"

    def test_model_health_tracking(self, sample_config):
        """测试模型健康状态跟踪."""
        router = ModelRouter(config_path=sample_config)

        # 标记失败
        router.mark_model_failed("gpt-4")
        assert not router._model_health["gpt-4"]["healthy"]
        assert router._model_health["gpt-4"]["fail_count"] == 1

        # 标记成功
        router.mark_model_success("gpt-4")
        assert router._model_health["gpt-4"]["healthy"]
        assert router._model_health["gpt-4"]["fail_count"] == 0

    def test_suggest_model(self, sample_config):
        """测试模型建议."""
        router = ModelRouter(config_path=sample_config)

        suggestions = router.suggest_models(
            task_type="coding",
            max_cost=0.01,
        )

        assert len(suggestions) > 0
        # 低成本任务应该建议 deepseek-coder
        assert any(s["name"] == "deepseek-coder" for s in suggestions)


class TestModelRouterEmpty:
    """测试空配置的模型路由器."""

    def test_no_config(self):
        """测试无配置文件."""
        router = ModelRouter(config_path=None)
        assert router._models == {}

    def test_get_model_no_config(self):
        """测试无配置时获取模型."""
        router = ModelRouter(config_path=None)
        model = router.get_model_for_task("coding")
        assert model is None
