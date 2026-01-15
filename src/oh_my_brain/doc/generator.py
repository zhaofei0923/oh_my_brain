"""开发文档生成器.

支持通过 LLM 自动生成开发文档，或验证用户手动添加的文档。
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from oh_my_brain.schemas.dev_doc import DevDoc, Module, ProjectInfo, SubTask, TaskType, TechStack

logger = logging.getLogger(__name__)


class ProjectType(str, Enum):
    """项目类型枚举."""

    WEB_API = "web_api"           # Web API / 后端服务
    WEB_FRONTEND = "web_frontend"  # Web 前端 (React/Vue/Angular)
    H5_MOBILE = "h5_mobile"        # H5 移动端
    SAAS_PLATFORM = "saas"         # SaaS 平台
    DATA_PLATFORM = "data"         # 数据平台 (ETL/数仓)
    WEBSITE = "website"            # 网站
    CLI_TOOL = "cli"               # 命令行工具
    LIBRARY = "library"            # 库/SDK
    CPP_ALGORITHM = "cpp_algo"     # C++ 算法
    FULLSTACK = "fullstack"        # 全栈应用
    MICROSERVICE = "microservice"  # 微服务架构


class GenerationMode(str, Enum):
    """文档生成模式."""

    AUTO = "auto"          # LLM 完全自动生成
    INTERACTIVE = "interactive"  # 交互式生成（用户确认每一步）
    MANUAL = "manual"      # 用户手动编写，系统只验证


# 项目类型到技术栈模板的映射
PROJECT_TEMPLATES: dict[ProjectType, dict[str, Any]] = {
    ProjectType.WEB_API: {
        "tech_stack": {
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "other": ["Redis", "Docker", "pytest"],
        },
        "common_modules": [
            "用户认证 (Authentication)",
            "权限管理 (Authorization)",
            "API 接口",
            "数据模型",
            "业务逻辑",
            "单元测试",
        ],
    },
    ProjectType.WEB_FRONTEND: {
        "tech_stack": {
            "language": "TypeScript",
            "framework": "React",
            "database": None,
            "other": ["Vite", "TailwindCSS", "React Query", "Vitest"],
        },
        "common_modules": [
            "页面路由",
            "状态管理",
            "UI 组件库",
            "API 集成",
            "表单处理",
            "单元测试",
        ],
    },
    ProjectType.H5_MOBILE: {
        "tech_stack": {
            "language": "TypeScript",
            "framework": "Vue3",
            "database": None,
            "other": ["Vant", "Vite", "Pinia"],
        },
        "common_modules": [
            "页面布局",
            "移动端适配",
            "手势交互",
            "本地存储",
            "API 对接",
        ],
    },
    ProjectType.SAAS_PLATFORM: {
        "tech_stack": {
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "other": ["Redis", "Celery", "Docker", "Kubernetes"],
        },
        "common_modules": [
            "多租户系统",
            "用户认证",
            "权限管理",
            "订阅计费",
            "管理后台",
            "API 网关",
            "监控告警",
        ],
    },
    ProjectType.DATA_PLATFORM: {
        "tech_stack": {
            "language": "Python",
            "framework": "Apache Airflow",
            "database": "ClickHouse",
            "other": ["Spark", "Kafka", "Flink", "dbt"],
        },
        "common_modules": [
            "数据采集",
            "ETL 流程",
            "数据建模",
            "数据质量",
            "报表分析",
            "调度管理",
        ],
    },
    ProjectType.WEBSITE: {
        "tech_stack": {
            "language": "TypeScript",
            "framework": "Next.js",
            "database": None,
            "other": ["TailwindCSS", "MDX", "Vercel"],
        },
        "common_modules": [
            "页面布局",
            "导航菜单",
            "内容管理",
            "SEO 优化",
            "响应式设计",
        ],
    },
    ProjectType.CLI_TOOL: {
        "tech_stack": {
            "language": "Python",
            "framework": "Typer",
            "database": None,
            "other": ["Rich", "pytest"],
        },
        "common_modules": [
            "命令解析",
            "配置管理",
            "输出格式化",
            "错误处理",
            "单元测试",
        ],
    },
    ProjectType.LIBRARY: {
        "tech_stack": {
            "language": "Python",
            "framework": None,
            "database": None,
            "other": ["pytest", "sphinx", "mypy"],
        },
        "common_modules": [
            "核心功能",
            "公共 API",
            "类型定义",
            "文档",
            "示例代码",
            "单元测试",
        ],
    },
    ProjectType.CPP_ALGORITHM: {
        "tech_stack": {
            "language": "C++",
            "framework": None,
            "database": None,
            "other": ["CMake", "GoogleTest", "Benchmark"],
        },
        "common_modules": [
            "算法实现",
            "数据结构",
            "性能优化",
            "接口封装",
            "单元测试",
            "性能测试",
        ],
    },
    ProjectType.FULLSTACK: {
        "tech_stack": {
            "language": "TypeScript",
            "framework": "Next.js",
            "database": "PostgreSQL",
            "other": ["Prisma", "TailwindCSS", "Docker"],
        },
        "common_modules": [
            "前端页面",
            "后端 API",
            "数据模型",
            "用户认证",
            "部署配置",
        ],
    },
    ProjectType.MICROSERVICE: {
        "tech_stack": {
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "other": ["gRPC", "Kafka", "Docker", "Kubernetes", "Consul"],
        },
        "common_modules": [
            "服务注册发现",
            "API 网关",
            "各业务服务",
            "消息队列",
            "分布式追踪",
            "配置中心",
        ],
    },
}


class DocGeneratorPrompts:
    """文档生成提示词模板."""

    SYSTEM_PROMPT = """你是一个专业的软件架构师和技术文档专家。
你的任务是根据用户的需求描述，生成标准化的开发文档。

开发文档必须严格遵循以下 YAML 格式：

```yaml
project:
  name: "项目名称"
  version: "0.1.0"
  description: "项目描述"
  tech_stack:
    language: "主要编程语言"
    framework: "使用的框架"
    database: "数据库（可选）"
    other:
      - "其他技术栈"

modules:
  - id: "mod-xxx"  # 模块ID，格式为 mod-小写字母数字
    name: "模块名称"
    description: "模块描述"
    priority: 1  # 优先级 1-5，1最高
    dependencies: []  # 依赖的模块ID列表
    acceptance_criteria: "验收标准"
    sub_tasks:
      - id: "task-001"  # 任务ID，格式为 task-三位数字
        name: "任务名称"
        type: "feature"  # feature/bugfix/refactor/test/docs
        description: "详细描述"
        estimated_minutes: 30  # 预估时长 5-120 分钟
        files_involved:
          - "src/xxx.py"
        requirements: "具体需求说明"
```

生成规则：
1. 每个模块应包含 3-8 个子任务
2. 任务粒度适中，每个任务 15-60 分钟可完成
3. 考虑模块间依赖关系
4. 优先级合理分配
5. 验收标准明确可测试
6. 文件路径要符合项目结构惯例
"""

    GENERATE_FROM_REQUIREMENTS = """请根据以下需求生成完整的开发文档：

## 项目类型
{project_type}

## 需求描述
{requirements}

## 技术栈建议
{tech_stack}

请生成符合格式要求的 YAML 开发文档。只输出 YAML 内容，不要包含其他解释。
"""

    GENERATE_MODULE = """请为以下模块生成子任务列表：

## 项目信息
- 项目名称: {project_name}
- 技术栈: {tech_stack}

## 模块信息
- 模块名称: {module_name}
- 模块描述: {module_description}
- 验收标准: {acceptance_criteria}

请生成该模块的子任务列表，格式为 YAML 数组。只输出 sub_tasks 部分。
"""

    REFINE_TASK = """请细化以下任务的需求：

## 任务信息
- 任务名称: {task_name}
- 当前描述: {description}
- 涉及文件: {files}

## 项目上下文
{context}

请生成更详细的 requirements 字段内容，包括：
1. 具体的功能要求
2. 边界条件处理
3. 错误处理要求
4. 代码规范要求
"""


class DocGenerator:
    """开发文档生成器.

    支持通过 LLM 自动生成或交互式生成开发文档。
    """

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4",
    ):
        """初始化生成器.

        Args:
            llm_client: LLM 客户端实例
            model: 使用的模型名称
        """
        self._llm_client = llm_client
        self._model = model
        self._prompts = DocGeneratorPrompts()

    async def generate_from_requirements(
        self,
        requirements: str,
        project_type: ProjectType = ProjectType.WEB_API,
        project_name: str | None = None,
        custom_tech_stack: dict[str, Any] | None = None,
    ) -> DevDoc:
        """从需求描述生成开发文档.

        Args:
            requirements: 需求描述文本
            project_type: 项目类型
            project_name: 项目名称（可选）
            custom_tech_stack: 自定义技术栈（可选）

        Returns:
            生成的开发文档
        """
        # 获取模板技术栈
        template = PROJECT_TEMPLATES.get(project_type, PROJECT_TEMPLATES[ProjectType.WEB_API])
        tech_stack = custom_tech_stack or template["tech_stack"]

        # 构建提示
        prompt = self._prompts.GENERATE_FROM_REQUIREMENTS.format(
            project_type=project_type.value,
            requirements=requirements,
            tech_stack=yaml.dump(tech_stack, allow_unicode=True),
        )

        # 调用 LLM
        if self._llm_client:
            response = await self._call_llm(prompt)
            yaml_content = self._extract_yaml(response)
            return self._parse_yaml_to_doc(yaml_content)
        else:
            # 无 LLM 时生成模板
            return self._generate_template(
                project_name or "my-project",
                requirements,
                project_type,
                tech_stack,
            )

    async def generate_module(
        self,
        project_info: ProjectInfo,
        module_name: str,
        module_description: str,
        acceptance_criteria: str,
    ) -> Module:
        """为单个模块生成子任务.

        Args:
            project_info: 项目信息
            module_name: 模块名称
            module_description: 模块描述
            acceptance_criteria: 验收标准

        Returns:
            生成的模块
        """
        prompt = self._prompts.GENERATE_MODULE.format(
            project_name=project_info.name,
            tech_stack=f"{project_info.tech_stack.language}/{project_info.tech_stack.framework}",
            module_name=module_name,
            module_description=module_description,
            acceptance_criteria=acceptance_criteria,
        )

        if self._llm_client:
            response = await self._call_llm(prompt)
            sub_tasks_yaml = self._extract_yaml(response)
            sub_tasks_data = yaml.safe_load(sub_tasks_yaml)

            sub_tasks = []
            for i, task_data in enumerate(sub_tasks_data):
                task_data.setdefault("id", f"task-{i+1:03d}")
                task_data.setdefault("type", "feature")
                task_data.setdefault("estimated_minutes", 30)
                sub_tasks.append(SubTask(**task_data))

            # 生成模块 ID
            module_id = f"mod-{module_name.lower().replace(' ', '-')[:20]}"

            return Module(
                id=module_id,
                name=module_name,
                description=module_description,
                acceptance_criteria=acceptance_criteria,
                sub_tasks=sub_tasks,
            )
        else:
            # 无 LLM 时生成占位模块
            return self._generate_placeholder_module(
                module_name,
                module_description,
                acceptance_criteria,
            )

    async def refine_task(
        self,
        task: SubTask,
        context: str = "",
    ) -> SubTask:
        """细化任务需求.

        Args:
            task: 原始任务
            context: 项目上下文

        Returns:
            细化后的任务
        """
        prompt = self._prompts.REFINE_TASK.format(
            task_name=task.name,
            description=task.description,
            files=", ".join(task.files_involved),
            context=context,
        )

        if self._llm_client:
            response = await self._call_llm(prompt)
            refined_requirements = response.strip()

            return SubTask(
                id=task.id,
                name=task.name,
                type=task.type,
                description=task.description,
                estimated_minutes=task.estimated_minutes,
                files_involved=task.files_involved,
                requirements=refined_requirements,
                preferred_model=task.preferred_model,
            )
        else:
            return task

    def create_from_template(
        self,
        project_name: str,
        project_type: ProjectType,
        description: str = "",
    ) -> DevDoc:
        """从模板创建开发文档骨架.

        Args:
            project_name: 项目名称
            project_type: 项目类型
            description: 项目描述

        Returns:
            文档骨架（需要用户填充详细内容）
        """
        template = PROJECT_TEMPLATES.get(project_type, PROJECT_TEMPLATES[ProjectType.WEB_API])
        tech_stack_data = template["tech_stack"]
        common_modules = template["common_modules"]

        # 创建技术栈
        tech_stack = TechStack(
            language=tech_stack_data["language"],
            framework=tech_stack_data.get("framework"),
            database=tech_stack_data.get("database"),
            other=tech_stack_data.get("other", []),
        )

        # 创建项目信息
        project_info = ProjectInfo(
            name=project_name,
            version="0.1.0",
            description=description or f"{project_type.value} 项目",
            tech_stack=tech_stack,
        )

        # 创建模块骨架
        modules = []
        for i, module_name in enumerate(common_modules):
            module_id = f"mod-{module_name.lower().replace(' ', '-').replace('(', '').replace(')', '')[:20]}"
            module_id = module_id.replace('--', '-').rstrip('-')

            # 确保 module_id 符合正则模式
            import re
            module_id = re.sub(r'[^a-z0-9-]', '', module_id)
            if not module_id.startswith("mod-"):
                module_id = f"mod-{module_id}"

            modules.append(Module(
                id=module_id,
                name=module_name,
                description=f"TODO: 填写 {module_name} 的详细描述",
                priority=min(i + 1, 5),
                dependencies=[modules[i-1].id] if i > 0 else [],
                acceptance_criteria=f"TODO: 填写 {module_name} 的验收标准",
                sub_tasks=[
                    SubTask(
                        id=f"task-{i+1:02d}1",
                        name=f"实现 {module_name} 核心功能",
                        type=TaskType.FEATURE,
                        description=f"TODO: 填写详细描述",
                        estimated_minutes=60,
                        files_involved=[],
                        requirements=f"TODO: 填写具体需求",
                    ),
                ],
            ))

        return DevDoc(project=project_info, modules=modules)

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM.

        Args:
            prompt: 提示词

        Returns:
            LLM 响应
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not configured")

        try:
            # 支持多种 LLM 客户端接口
            if hasattr(self._llm_client, "chat"):
                # OpenAI 风格
                response = await self._llm_client.chat(
                    messages=[
                        {"role": "system", "content": self._prompts.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=self._model,
                )
                return response.choices[0].message.content
            elif hasattr(self._llm_client, "generate"):
                # 通用接口
                return await self._llm_client.generate(
                    system_prompt=self._prompts.SYSTEM_PROMPT,
                    prompt=prompt,
                    model=self._model,
                )
            else:
                raise RuntimeError("Unsupported LLM client interface")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _extract_yaml(self, response: str) -> str:
        """从 LLM 响应中提取 YAML 内容.

        Args:
            response: LLM 响应

        Returns:
            YAML 内容
        """
        # 尝试提取代码块
        if "```yaml" in response:
            start = response.find("```yaml") + 7
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                content = response[start:end].strip()
                # 跳过语言标识
                if content.startswith("yaml") or content.startswith("yml"):
                    content = content[4:].strip()
                return content

        # 直接返回（假设整个响应就是 YAML）
        return response.strip()

    def _parse_yaml_to_doc(self, yaml_content: str) -> DevDoc:
        """解析 YAML 为 DevDoc.

        Args:
            yaml_content: YAML 内容

        Returns:
            DevDoc 对象
        """
        data = yaml.safe_load(yaml_content)
        return DevDoc(**data)

    def _generate_template(
        self,
        project_name: str,
        requirements: str,
        project_type: ProjectType,
        tech_stack: dict[str, Any],
    ) -> DevDoc:
        """生成模板文档（无 LLM 时使用）."""
        return self.create_from_template(project_name, project_type, requirements[:200])

    def _generate_placeholder_module(
        self,
        module_name: str,
        description: str,
        acceptance_criteria: str,
    ) -> Module:
        """生成占位模块."""
        import re
        module_id = f"mod-{module_name.lower().replace(' ', '-')[:20]}"
        module_id = re.sub(r'[^a-z0-9-]', '', module_id)

        return Module(
            id=module_id,
            name=module_name,
            description=description,
            acceptance_criteria=acceptance_criteria,
            sub_tasks=[
                SubTask(
                    id="task-001",
                    name=f"实现 {module_name}",
                    type=TaskType.FEATURE,
                    description=description,
                    estimated_minutes=60,
                    files_involved=[],
                    requirements="TODO: 需要使用 LLM 生成详细需求",
                ),
            ],
        )


def save_dev_doc(doc: DevDoc, path: Path | str) -> None:
    """保存开发文档到文件.

    Args:
        doc: 开发文档
        path: 文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为字典
    data = doc.model_dump(mode="json")

    # 根据扩展名选择格式
    if path.suffix in [".yaml", ".yml"]:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Dev doc saved to {path}")


def load_dev_doc(path: Path | str) -> DevDoc:
    """从文件加载开发文档.

    Args:
        path: 文件路径

    Returns:
        开发文档
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return DevDoc(**data)
