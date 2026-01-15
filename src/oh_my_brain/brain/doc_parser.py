"""开发文档解析器."""

import logging
from pathlib import Path

import yaml

from oh_my_brain.schemas.dev_doc import DevDoc

logger = logging.getLogger(__name__)


class DocParser:
    """开发文档解析器.

    解析用户上传的YAML开发文档。
    """

    @staticmethod
    def parse_file(path: Path | str) -> DevDoc:
        """解析YAML文件.

        Args:
            path: 文件路径

        Returns:
            解析后的DevDoc对象

        Raises:
            ValueError: 文件格式错误或验证失败
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return DocParser.parse_dict(data)

    @staticmethod
    def parse_dict(data: dict) -> DevDoc:
        """解析字典数据.

        Args:
            data: 字典数据

        Returns:
            解析后的DevDoc对象

        Raises:
            ValueError: 验证失败
        """
        try:
            return DevDoc.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid dev doc format: {e}")

    @staticmethod
    def parse_yaml_string(content: str) -> DevDoc:
        """解析YAML字符串.

        Args:
            content: YAML字符串

        Returns:
            解析后的DevDoc对象
        """
        data = yaml.safe_load(content)
        return DocParser.parse_dict(data)

    @staticmethod
    def validate_file(path: Path | str) -> tuple[bool, str]:
        """验证文件格式.

        Args:
            path: 文件路径

        Returns:
            (是否有效, 错误信息)
        """
        try:
            DocParser.parse_file(path)
            return True, ""
        except Exception as e:
            return False, str(e)

    @staticmethod
    def export_json_schema() -> dict:
        """导出JSON Schema.

        Returns:
            JSON Schema字典，可用于外部工具验证
        """
        return DevDoc.model_json_schema()

    @staticmethod
    def generate_template() -> str:
        """生成开发文档模板.

        Returns:
            YAML格式的模板字符串
        """
        template = """# OH MY BRAIN 开发文档模板
# 使用任意AI（Claude、DeepSeek、GPT等）填充此模板

project:
  name: "项目名称"
  version: "0.1.0"
  description: "项目描述"
  tech_stack:
    language: "Python"           # 主要编程语言
    framework: "FastAPI"         # 框架（可选）
    database: "PostgreSQL"       # 数据库（可选）
    other:                       # 其他技术（可选）
      - "Redis"
      - "Docker"

modules:
  # 模块1: 示例 - 用户认证
  - id: "mod-auth"               # 模块ID，格式: mod-xxx
    name: "用户认证模块"
    description: "处理用户注册、登录、JWT认证"
    priority: 1                  # 优先级 1-5，1最高
    dependencies: []             # 依赖的模块ID列表
    acceptance_criteria: "用户可以注册、登录、登出，JWT认证正常工作"

    sub_tasks:
      # 子任务1
      - id: "task-001"           # 任务ID，格式: task-xxx
        name: "实现用户注册"
        type: "feature"          # feature/bugfix/refactor/test/docs
        description: "创建用户注册接口，包含邮箱验证"
        estimated_minutes: 30    # 预估时长 5-120 分钟
        files_involved:          # 涉及的文件
          - "src/auth/router.py"
          - "src/auth/service.py"
          - "src/auth/schemas.py"
        requirements: |
          - POST /auth/register 接口
          - 请求体: email, password, name
          - 密码强度验证（至少8位，包含数字和字母）
          - 邮箱格式验证
          - 返回用户信息（不含密码）
        # preferred_model: "claude-opus"  # 可选：指定使用的AI模型

      # 子任务2
      - id: "task-002"
        name: "实现JWT登录"
        type: "feature"
        description: "创建登录接口，生成JWT令牌"
        estimated_minutes: 30
        files_involved:
          - "src/auth/router.py"
          - "src/auth/service.py"
        requirements: |
          - POST /auth/login 接口
          - 请求体: email, password
          - 验证用户凭据
          - 成功返回 access_token 和 refresh_token

  # 模块2: 示例 - 数据模型（依赖认证模块）
  - id: "mod-models"
    name: "数据模型模块"
    description: "定义数据库模型和迁移"
    priority: 2
    dependencies:
      - "mod-auth"               # 依赖认证模块完成后再开始
    acceptance_criteria: "所有数据模型定义完成，迁移可正常执行"

    sub_tasks:
      - id: "task-003"
        name: "定义User模型"
        type: "feature"
        description: "创建用户数据库模型"
        estimated_minutes: 20
        files_involved:
          - "src/models/user.py"
        requirements: |
          - 使用 SQLAlchemy ORM
          - 字段: id, email, hashed_password, name, created_at
          - 邮箱唯一索引
"""
        return template
