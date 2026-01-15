"""Mini-Agent适配器.

适配Mini-Agent的工具系统，添加安全检查和上下文管理。
"""

import asyncio
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oh_my_brain.schemas.config import WorkerConfig
from oh_my_brain.schemas.task import Task, TaskResult, TaskStatus
from oh_my_brain.worker.base import WorkerBase
from oh_my_brain.worker.brain_client import BrainClient

logger = logging.getLogger(__name__)


class ToolInterceptor:
    """工具拦截器.

    拦截Mini-Agent的工具调用，添加安全检查。
    """

    def __init__(self, brain_client: BrainClient):
        self._brain_client = brain_client

    async def intercept_bash(
        self,
        original_func: Callable,
        command: str,
        **kwargs,
    ) -> Any:
        """拦截bash命令.

        Args:
            original_func: 原始函数
            command: 命令
            **kwargs: 其他参数

        Returns:
            执行结果
        """
        # 安全检查
        approved, reason = await self._brain_client.request_safety_check(
            command_type="bash",
            command=command,
        )

        if not approved:
            logger.warning(f"Bash command rejected: {reason}")
            return {"error": f"Command rejected: {reason}"}

        # 执行原始命令
        return await original_func(command, **kwargs)

    async def intercept_write(
        self,
        original_func: Callable,
        file_path: str,
        content: str,
        **kwargs,
    ) -> Any:
        """拦截文件写入.

        Args:
            original_func: 原始函数
            file_path: 文件路径
            content: 文件内容
            **kwargs: 其他参数

        Returns:
            执行结果
        """
        # 安全检查
        approved, reason = await self._brain_client.request_safety_check(
            command_type="write",
            command=file_path,
            args={"content_length": len(content)},
        )

        if not approved:
            logger.warning(f"Write rejected: {reason}")
            return {"error": f"Write rejected: {reason}"}

        # 执行原始操作
        return await original_func(file_path, content, **kwargs)

    async def intercept_edit(
        self,
        original_func: Callable,
        file_path: str,
        old_text: str,
        new_text: str,
        **kwargs,
    ) -> Any:
        """拦截文件编辑.

        Args:
            original_func: 原始函数
            file_path: 文件路径
            old_text: 旧文本
            new_text: 新文本
            **kwargs: 其他参数

        Returns:
            执行结果
        """
        # 安全检查
        approved, reason = await self._brain_client.request_safety_check(
            command_type="edit",
            command=file_path,
            args={"old_text_length": len(old_text), "new_text_length": len(new_text)},
        )

        if not approved:
            logger.warning(f"Edit rejected: {reason}")
            return {"error": f"Edit rejected: {reason}"}

        # 执行原始操作
        return await original_func(file_path, old_text, new_text, **kwargs)


class MiniAgentAdapter(WorkerBase):
    """Mini-Agent适配器.

    将Mini-Agent作为执行后端，添加：
    1. 安全检查
    2. 上下文管理
    3. 任务进度上报
    """

    def __init__(
        self,
        config: WorkerConfig,
        mini_agent_path: Path | str | None = None,
    ):
        super().__init__(config)
        self._mini_agent_path = Path(mini_agent_path) if mini_agent_path else None
        self._tool_interceptor = ToolInterceptor(self._brain_client)
        self._mini_agent_process: subprocess.Popen | None = None

        # 设置能力
        self.set_capabilities(["python", "bash", "file_operations"])

    async def execute_task(self, task: Task) -> TaskResult:
        """执行任务.

        使用Mini-Agent执行任务。

        Args:
            task: 任务对象

        Returns:
            任务结果
        """
        logger.info(f"Executing task with Mini-Agent: {task.id}")

        try:
            # 获取任务上下文
            context = await self.request_context(["project_root", "git_branch"])

            # 构建Mini-Agent提示
            prompt = self._build_prompt(task, context)

            # 获取模型配置
            model_config = task.model_config or {}

            # 调用Mini-Agent
            result = await self._run_mini_agent(
                prompt=prompt,
                model=model_config.get("model_name", "claude-3-opus-20240229"),
                working_dir=context.get("project_root", "."),
            )

            # 解析结果
            if result.get("success"):
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    output=result.get("output", ""),
                    files_modified=result.get("files_modified", []),
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error_message=result.get("error", "Unknown error"),
                )

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error_message=str(e),
            )

    def _build_prompt(self, task: Task, context: dict[str, Any]) -> str:
        """构建Mini-Agent提示.

        Args:
            task: 任务对象
            context: 上下文

        Returns:
            提示字符串
        """
        prompt_parts = [
            f"# Task: {task.name}",
            "",
            "## Description",
            task.description or "",
            "",
        ]

        if task.requirements:
            prompt_parts.extend(
                [
                    "## Requirements",
                    task.requirements,
                    "",
                ]
            )

        if task.files_involved:
            prompt_parts.extend(
                [
                    "## Files to work on",
                    "\n".join(f"- {f}" for f in task.files_involved),
                    "",
                ]
            )

        if task.acceptance_criteria:
            prompt_parts.extend(
                [
                    "## Acceptance Criteria",
                    task.acceptance_criteria,
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Instructions",
                "Complete the task as described above.",
                "Make sure to follow best practices and write clean, maintainable code.",
                "Test your changes before marking the task as complete.",
            ]
        )

        return "\n".join(prompt_parts)

    async def _run_mini_agent(
        self,
        prompt: str,
        model: str,
        working_dir: str,
    ) -> dict[str, Any]:
        """运行Mini-Agent.

        Args:
            prompt: 提示
            model: 模型名称
            working_dir: 工作目录

        Returns:
            执行结果
        """
        # 方式1: 使用subprocess调用mini-agent CLI
        if self._mini_agent_path and self._mini_agent_path.exists():
            return await self._run_via_subprocess(prompt, model, working_dir)

        # 方式2: 直接导入并调用
        return await self._run_via_import(prompt, model, working_dir)

    async def _run_via_subprocess(
        self,
        prompt: str,
        model: str,
        working_dir: str,
    ) -> dict[str, Any]:
        """通过subprocess运行Mini-Agent."""
        cmd = [
            sys.executable,
            "-m",
            "mini_agent",
            "--model",
            model,
            "--working-dir",
            working_dir,
            "--prompt",
            prompt,
            "--non-interactive",
        ]

        try:
            # 异步运行
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {
                    "success": True,
                    "output": stdout.decode(),
                    "files_modified": [],  # TODO: 解析输出获取修改的文件
                }
            else:
                return {
                    "success": False,
                    "error": stderr.decode() or "Mini-Agent execution failed",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _run_via_import(
        self,
        prompt: str,
        model: str,
        working_dir: str,
    ) -> dict[str, Any]:
        """通过导入运行Mini-Agent.

        尝试导入mini_agent模块并直接调用。
        这需要mini_agent已安装在当前环境中。
        """
        try:
            # 尝试导入 - 具体实现取决于mini-agent的API
            # 这里提供一个通用的接口适配

            # 检查是否安装了mini-agent
            import importlib.util

            spec = importlib.util.find_spec("mini_agent")
            if spec is None:
                return {
                    "success": False,
                    "error": "mini_agent not installed. Install with: pip install mini-agent",
                }

            # TODO: 根据mini-agent的实际API实现
            # 当前返回占位符结果
            logger.warning("Direct mini-agent import not yet implemented")
            return {
                "success": False,
                "error": "Direct import not implemented, use subprocess mode",
            }

        except ImportError as e:
            return {
                "success": False,
                "error": f"Failed to import mini_agent: {e}",
            }

    async def setup_tool_interceptors(self) -> None:
        """设置工具拦截器.

        注入安全检查到Mini-Agent的工具中。
        """
        # TODO: 实现工具拦截
        # 需要根据mini-agent的工具系统实现
        pass


class SimpleMiniAgentWorker(MiniAgentAdapter):
    """简化版Mini-Agent Worker.

    不依赖mini-agent包，使用内置的简单工具实现。
    适合测试和简单任务。
    """

    async def _run_via_import(
        self,
        prompt: str,
        model: str,
        working_dir: str,
    ) -> dict[str, Any]:
        """使用内置实现运行任务."""
        # 使用简单的LLM调用实现
        # 这里可以直接调用配置的AI API

        try:
            # 尝试使用配置的模型
            from oh_my_brain.worker.llm_client import LLMClient

            client = LLMClient(model_name=model)
            response = await client.complete(prompt)

            return {
                "success": True,
                "output": response,
                "files_modified": [],
            }

        except ImportError:
            # LLMClient还未实现，返回占位符
            return {
                "success": False,
                "error": "LLM client not yet implemented",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
