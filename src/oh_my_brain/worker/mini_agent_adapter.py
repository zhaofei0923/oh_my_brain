"""Mini-Agent适配器.

适配Mini-Agent的工具系统，添加安全检查和上下文管理。

支持两种模式：
1. 导入模式：直接导入 mini_agent 模块（需要安装 mini-agent 包）
2. subprocess模式：通过CLI调用 mini-agent
3. 内置模式：使用内置的简化LLM客户端（不依赖 mini-agent）
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oh_my_brain.schemas.config import WorkerConfig
from oh_my_brain.schemas.task import Task, TaskResult
from oh_my_brain.worker.base import WorkerBase
from oh_my_brain.worker.brain_client import BrainClient

logger = logging.getLogger(__name__)

# Mini-Agent 是否可用
MINI_AGENT_AVAILABLE = False
try:
    from mini_agent.agent import Agent as MiniAgent
    from mini_agent.llm import LLMClient as MiniAgentLLMClient
    from mini_agent.schema import Message as MiniAgentMessage
    from mini_agent.tools.base import Tool as MiniAgentTool, ToolResult as MiniAgentToolResult
    MINI_AGENT_AVAILABLE = True
except ImportError:
    MiniAgent = None
    MiniAgentLLMClient = None
    MiniAgentMessage = None
    MiniAgentTool = None
    MiniAgentToolResult = None


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
            model_name: str = model_config.get("model_name", "claude-3-opus-20240229")
            result = await self._run_mini_agent(
                prompt=prompt,
                model=model_name,
                working_dir=context.get("project_root", "."),
            )

            # 解析结果
            if result.get("success"):
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    output=result.get("output", ""),
                    files_modified=result.get("files_modified", []),
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error=result.get("error", "Unknown error"),
                )

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
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

        prompt_parts.extend(
            [
                "## Instructions",
                "Complete the task as described above.",
                "Make sure to follow best practices and write clean, maintainable code.",
                "Test your changes before marking the task as complete.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_modified_files(self, output: str) -> list[str]:
        """从 Mini-Agent 输出中解析修改的文件.

        Args:
            output: Mini-Agent 输出

        Returns:
            修改的文件路径列表
        """
        import re

        files_modified = []

        # 匹配常见的文件操作模式
        patterns = [
            # Created/Modified file: xxx
            r"(?:Created|Modified|Updated|Wrote|Writing)\s+(?:file[:\s]+)?([^\s\n]+\.[a-zA-Z0-9]+)",
            # ✓ file.py
            r"[✓✔]\s+([^\s\n]+\.[a-zA-Z0-9]+)",
            # File: path/to/file.py
            r"File:\s*([^\s\n]+\.[a-zA-Z0-9]+)",
            # edit_file tool usage patterns
            r"edit_file.*?path[\"']?\s*[:\s]+\s*[\"']?([^\s\n\"']+)[\"']?",
            # create_file tool usage patterns
            r"create_file.*?path[\"']?\s*[:\s]+\s*[\"']?([^\s\n\"']+)[\"']?",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                # 清理路径
                path = match.strip().strip("'\"")
                if path and path not in files_modified:
                    # 过滤掉明显不是文件的匹配
                    if not path.startswith("http") and "." in path:
                        files_modified.append(path)

        return files_modified

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
                # 解析输出获取修改的文件
                files_modified = self._parse_modified_files(stdout.decode())
                return {
                    "success": True,
                    "output": stdout.decode(),
                    "files_modified": files_modified,
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

        直接导入mini_agent模块并调用Agent执行任务。
        需要mini_agent已安装在当前环境中。
        """
        if not MINI_AGENT_AVAILABLE:
            return {
                "success": False,
                "error": "mini_agent not installed. Install with: pip install 'oh-my-brain[mini-agent]'",
            }

        try:
            # 配置 Mini-Agent
            api_key = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("MINIMAX_API_BASE", "https://api.minimax.io")

            if not api_key:
                return {
                    "success": False,
                    "error": "No API key found. Set MINIMAX_API_KEY or OPENAI_API_KEY environment variable.",
                }

            # 创建 LLM 客户端
            llm_client = MiniAgentLLMClient(
                api_key=api_key,
                api_base=api_base,
                model=model,
            )

            # 加载默认工具
            tools = await self._load_mini_agent_tools(working_dir)

            # 创建系统提示
            system_prompt = self._build_system_prompt(working_dir)

            # 创建 Agent
            agent = MiniAgent(
                llm_client=llm_client,
                system_prompt=system_prompt,
                tools=tools,
                max_steps=50,
                workspace_dir=working_dir,
            )

            # 添加用户任务
            agent.add_user_message(prompt)

            # 执行任务
            logger.info("Starting Mini-Agent execution...")
            result = await agent.run()

            # 解析输出，提取修改的文件
            files_modified = self._parse_modified_files(agent.get_history())

            return {
                "success": True,
                "output": result,
                "files_modified": files_modified,
            }

        except Exception as e:
            logger.error(f"Mini-Agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _load_mini_agent_tools(self, working_dir: str) -> list:
        """加载 Mini-Agent 工具并注入安全检查.

        Args:
            working_dir: 工作目录

        Returns:
            工具列表
        """
        if not MINI_AGENT_AVAILABLE:
            return []

        tools = []

        try:
            # 导入默认工具
            from mini_agent.tools.bash import BashTool
            from mini_agent.tools.file import (
                CreateFileTool,
                EditFileTool,
                ReadFileTool,
            )

            # 创建安全包装的工具
            bash_tool = SafeBashTool(
                brain_client=self._brain_client,
                working_dir=working_dir,
            )
            tools.append(bash_tool)

            # 文件工具
            read_file = ReadFileTool(workspace_dir=working_dir)
            create_file = SafeCreateFileTool(
                brain_client=self._brain_client,
                working_dir=working_dir,
            )
            edit_file = SafeEditFileTool(
                brain_client=self._brain_client,
                working_dir=working_dir,
            )

            tools.extend([read_file, create_file, edit_file])

            logger.info(f"Loaded {len(tools)} Mini-Agent tools")

        except ImportError as e:
            logger.warning(f"Failed to import Mini-Agent tools: {e}")
            # 返回空列表，使用 Mini-Agent 的默认工具

        return tools

    def _build_system_prompt(self, working_dir: str) -> str:
        """构建系统提示."""
        return f"""You are an AI coding assistant working on a software development task.

## Current Workspace
You are working in: `{working_dir}`
All file paths should be relative to this directory.

## Guidelines
1. Read existing files before making changes to understand the codebase
2. Write clean, maintainable code following best practices
3. Test your changes when possible
4. Commit your work with clear commit messages
5. If you encounter errors, debug and fix them

## Safety Rules
- Do not delete system files or directories
- Do not run destructive commands without explicit approval
- Always verify file paths before writing
"""

    def _parse_modified_files(self, history: list) -> list[str]:
        """从历史记录中解析修改的文件."""
        files = set()

        for msg in history:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = tool_call.function.arguments

                    # 检测文件操作
                    if func_name in ('create_file', 'edit_file', 'write_file'):
                        if 'file_path' in args:
                            files.add(args['file_path'])
                        elif 'path' in args:
                            files.add(args['path'])

        return list(files)

    async def setup_tool_interceptors(self) -> None:
        """设置工具拦截器.

        注入安全检查到Mini-Agent的工具中。
        工具拦截器在 _load_mini_agent_tools 中自动设置。
        """
        logger.info("Tool interceptors are configured during tool loading")
        # 拦截器通过安全工具包装类实现
        # 见: SafeBashTool, SafeCreateFileTool, SafeEditFileTool


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


# ============================================================
# 安全工具包装类
# ============================================================


class SafeBashTool:
    """安全的 Bash 工具包装.

    包装 Mini-Agent 的 BashTool，添加 Brain 安全检查。
    """

    name = "bash"
    description = "Execute a bash command in the workspace. Commands are subject to safety review."

    def __init__(self, brain_client: BrainClient, working_dir: str):
        self._brain_client = brain_client
        self._working_dir = working_dir
        self._original_tool = None

        # 尝试加载原始工具
        if MINI_AGENT_AVAILABLE:
            try:
                from mini_agent.tools.bash import BashTool
                self._original_tool = BashTool(workspace_dir=working_dir)
            except ImportError:
                pass

    @property
    def parameters(self) -> dict:
        """工具参数定义."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        }

    async def execute(self, command: str) -> Any:
        """执行 bash 命令（带安全检查）."""
        # 请求安全检查
        try:
            approved, reason = await self._brain_client.request_safety_check(
                command_type="bash",
                command=command,
            )

            if not approved:
                logger.warning(f"Bash command rejected: {reason}")
                if MINI_AGENT_AVAILABLE:
                    return MiniAgentToolResult(
                        success=False,
                        content="",
                        error=f"Command rejected by safety check: {reason}",
                    )
                return {"success": False, "error": f"Command rejected: {reason}"}

        except Exception as e:
            logger.warning(f"Safety check failed, proceeding with caution: {e}")

        # 使用原始工具执行
        if self._original_tool:
            return await self._original_tool.execute(command=command)

        # 如果没有原始工具，直接执行
        return await self._execute_command(command)

    async def _execute_command(self, command: str) -> Any:
        """直接执行命令."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )
            stdout, stderr = await process.communicate()

            output = stdout.decode() if stdout else ""
            error = stderr.decode() if stderr else ""

            if process.returncode == 0:
                if MINI_AGENT_AVAILABLE:
                    return MiniAgentToolResult(success=True, content=output)
                return {"success": True, "output": output}
            else:
                if MINI_AGENT_AVAILABLE:
                    return MiniAgentToolResult(
                        success=False,
                        content=output,
                        error=error or f"Command failed with code {process.returncode}",
                    )
                return {"success": False, "output": output, "error": error}

        except Exception as e:
            if MINI_AGENT_AVAILABLE:
                return MiniAgentToolResult(success=False, content="", error=str(e))
            return {"success": False, "error": str(e)}


class SafeCreateFileTool:
    """安全的文件创建工具包装."""

    name = "create_file"
    description = "Create a new file with the specified content."

    def __init__(self, brain_client: BrainClient, working_dir: str):
        self._brain_client = brain_client
        self._working_dir = working_dir
        self._original_tool = None

        if MINI_AGENT_AVAILABLE:
            try:
                from mini_agent.tools.file import CreateFileTool
                self._original_tool = CreateFileTool(workspace_dir=working_dir)
            except ImportError:
                pass

    @property
    def parameters(self) -> dict:
        """工具参数定义."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to create",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, file_path: str, content: str) -> Any:
        """创建文件（带安全检查）."""
        # 请求安全检查
        try:
            approved, reason = await self._brain_client.request_safety_check(
                command_type="write",
                command=file_path,
                args={"content_length": len(content)},
            )

            if not approved:
                logger.warning(f"File creation rejected: {reason}")
                if MINI_AGENT_AVAILABLE:
                    return MiniAgentToolResult(
                        success=False,
                        content="",
                        error=f"File creation rejected: {reason}",
                    )
                return {"success": False, "error": f"File creation rejected: {reason}"}

        except Exception as e:
            logger.warning(f"Safety check failed, proceeding with caution: {e}")

        # 使用原始工具
        if self._original_tool:
            return await self._original_tool.execute(file_path=file_path, content=content)

        # 直接创建文件
        return await self._create_file(file_path, content)

    async def _create_file(self, file_path: str, content: str) -> Any:
        """直接创建文件."""
        try:
            full_path = Path(self._working_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

            if MINI_AGENT_AVAILABLE:
                return MiniAgentToolResult(
                    success=True,
                    content=f"File created: {file_path}",
                )
            return {"success": True, "message": f"File created: {file_path}"}

        except Exception as e:
            if MINI_AGENT_AVAILABLE:
                return MiniAgentToolResult(success=False, content="", error=str(e))
            return {"success": False, "error": str(e)}


class SafeEditFileTool:
    """安全的文件编辑工具包装."""

    name = "edit_file"
    description = "Edit an existing file by replacing old text with new text."

    def __init__(self, brain_client: BrainClient, working_dir: str):
        self._brain_client = brain_client
        self._working_dir = working_dir
        self._original_tool = None

        if MINI_AGENT_AVAILABLE:
            try:
                from mini_agent.tools.file import EditFileTool
                self._original_tool = EditFileTool(workspace_dir=working_dir)
            except ImportError:
                pass

    @property
    def parameters(self) -> dict:
        """工具参数定义."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to be replaced",
                },
                "new_text": {
                    "type": "string",
                    "description": "New text to replace with",
                },
            },
            "required": ["file_path", "old_text", "new_text"],
        }

    async def execute(self, file_path: str, old_text: str, new_text: str) -> Any:
        """编辑文件（带安全检查）."""
        # 请求安全检查
        try:
            approved, reason = await self._brain_client.request_safety_check(
                command_type="edit",
                command=file_path,
                args={"old_text_length": len(old_text), "new_text_length": len(new_text)},
            )

            if not approved:
                logger.warning(f"File edit rejected: {reason}")
                if MINI_AGENT_AVAILABLE:
                    return MiniAgentToolResult(
                        success=False,
                        content="",
                        error=f"File edit rejected: {reason}",
                    )
                return {"success": False, "error": f"File edit rejected: {reason}"}

        except Exception as e:
            logger.warning(f"Safety check failed, proceeding with caution: {e}")

        # 使用原始工具
        if self._original_tool:
            return await self._original_tool.execute(
                file_path=file_path,
                old_text=old_text,
                new_text=new_text,
            )

        # 直接编辑文件
        return await self._edit_file(file_path, old_text, new_text)

    async def _edit_file(self, file_path: str, old_text: str, new_text: str) -> Any:
        """直接编辑文件."""
        try:
            full_path = Path(self._working_dir) / file_path

            if not full_path.exists():
                error = f"File not found: {file_path}"
                if MINI_AGENT_AVAILABLE:
                    return MiniAgentToolResult(success=False, content="", error=error)
                return {"success": False, "error": error}

            content = full_path.read_text(encoding="utf-8")

            if old_text not in content:
                error = "Old text not found in file"
                if MINI_AGENT_AVAILABLE:
                    return MiniAgentToolResult(success=False, content="", error=error)
                return {"success": False, "error": error}

            new_content = content.replace(old_text, new_text, 1)
            full_path.write_text(new_content, encoding="utf-8")

            if MINI_AGENT_AVAILABLE:
                return MiniAgentToolResult(
                    success=True,
                    content=f"File edited: {file_path}",
                )
            return {"success": True, "message": f"File edited: {file_path}"}

        except Exception as e:
            if MINI_AGENT_AVAILABLE:
                return MiniAgentToolResult(success=False, content="", error=str(e))
            return {"success": False, "error": str(e)}
