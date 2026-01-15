"""开发文档数据模型.

定义标准化开发文档的Schema，用户可使用任意AI生成符合此格式的开发文档。
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """任务类型枚举."""

    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"


class SubTask(BaseModel):
    """子任务定义.

    表示模块内的一个可独立执行的开发任务。
    """

    id: str = Field(..., description="子任务唯一ID，如 task-001", pattern=r"^task-\d{3,}$")
    name: str = Field(..., description="子任务名称", min_length=1, max_length=100)
    type: TaskType = Field(..., description="任务类型")
    description: str = Field(..., description="详细描述")
    estimated_minutes: int = Field(30, ge=5, le=120, description="预估时长(分钟)，建议5-120分钟")
    files_involved: list[str] = Field(default_factory=list, description="涉及的文件路径列表")
    requirements: str = Field(..., description="具体需求说明，越详细越好")
    preferred_model: str | None = Field(
        None, description="用户指定的AI模型名称（可选，覆盖默认映射）"
    )

    model_config = {"extra": "forbid"}


class Module(BaseModel):
    """模块定义.

    表示项目中的一个功能模块，可包含多个子任务。
    """

    id: str = Field(..., description="模块唯一ID，如 mod-auth", pattern=r"^mod-[a-z0-9-]+$")
    name: str = Field(..., description="模块名称", min_length=1, max_length=100)
    description: str = Field(..., description="模块描述")
    priority: int = Field(3, ge=1, le=5, description="优先级 1-5，1最高")
    dependencies: list[str] = Field(
        default_factory=list, description="依赖的模块ID列表，用于确定执行顺序"
    )
    acceptance_criteria: str = Field(..., description="验收标准，用于判断模块是否完成")
    sub_tasks: list[SubTask] = Field(..., min_length=1, description="子任务列表")

    model_config = {"extra": "forbid"}


class TechStack(BaseModel):
    """技术栈定义."""

    language: str = Field(..., description="主要编程语言，如 Python, TypeScript")
    framework: str | None = Field(None, description="使用的框架，如 FastAPI, React")
    database: str | None = Field(None, description="数据库，如 PostgreSQL, MongoDB")
    other: list[str] = Field(default_factory=list, description="其他技术，如 Redis, Docker")

    model_config = {"extra": "allow"}


class ProjectInfo(BaseModel):
    """项目信息."""

    name: str = Field(..., description="项目名称", min_length=1, max_length=100)
    version: str = Field("0.1.0", description="版本号", pattern=r"^\d+\.\d+\.\d+.*$")
    description: str = Field(..., description="项目描述")
    tech_stack: TechStack = Field(..., description="技术栈")
    repository: str | None = Field(None, description="Git仓库地址")

    model_config = {"extra": "forbid"}


class DevDoc(BaseModel):
    """开发文档根模型.

    用户上传的标准化开发文档，Brain据此分解任务并分配给Worker。

    Example:
        ```yaml
        project:
          name: "my-app"
          version: "0.1.0"
          description: "示例应用"
          tech_stack:
            language: "Python"
            framework: "FastAPI"
        modules:
          - id: "mod-auth"
            name: "认证模块"
            ...
        ```
    """

    project: ProjectInfo = Field(..., description="项目基本信息")
    modules: list[Module] = Field(..., min_length=1, description="模块列表")

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "project": {
                        "name": "my-awesome-app",
                        "version": "0.1.0",
                        "description": "一个示例Web应用",
                        "tech_stack": {
                            "language": "Python",
                            "framework": "FastAPI",
                            "database": "PostgreSQL",
                        },
                    },
                    "modules": [
                        {
                            "id": "mod-auth",
                            "name": "用户认证模块",
                            "description": "处理用户注册、登录、JWT认证",
                            "priority": 1,
                            "dependencies": [],
                            "acceptance_criteria": "用户可以注册、登录、登出",
                            "sub_tasks": [
                                {
                                    "id": "task-001",
                                    "name": "实现JWT登录",
                                    "type": "feature",
                                    "description": "创建登录接口",
                                    "estimated_minutes": 30,
                                    "files_involved": ["src/auth/router.py"],
                                    "requirements": "POST /auth/login 接口",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    }

    def get_task_dag(self) -> dict[str, list[str]]:
        """获取任务依赖DAG.

        Returns:
            模块ID到其依赖模块ID列表的映射
        """
        return {module.id: module.dependencies for module in self.modules}

    def get_all_tasks(self) -> list[tuple[str, SubTask]]:
        """获取所有子任务.

        Returns:
            (模块ID, 子任务) 元组列表
        """
        tasks = []
        for module in self.modules:
            for sub_task in module.sub_tasks:
                tasks.append((module.id, sub_task))
        return tasks

    def to_json_schema(self) -> dict[str, Any]:
        """导出JSON Schema供外部工具验证."""
        return self.model_json_schema()
