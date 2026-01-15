"""开发文档验证器.

提供详细的格式验证和错误提示，支持用户手动添加的文档审查。
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from oh_my_brain.schemas.dev_doc import DevDoc, Module, SubTask

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """验证问题严重程度."""

    ERROR = "error"      # 错误，必须修复
    WARNING = "warning"  # 警告，建议修复
    INFO = "info"        # 信息，可选优化


@dataclass
class ValidationIssue:
    """验证问题."""

    severity: ValidationSeverity
    path: str  # 问题位置，如 "modules[0].sub_tasks[1].id"
    message: str
    suggestion: str = ""
    code: str = ""  # 错误代码，如 "E001"

    def __str__(self) -> str:
        prefix = {
            ValidationSeverity.ERROR: "❌ ERROR",
            ValidationSeverity.WARNING: "⚠️  WARNING",
            ValidationSeverity.INFO: "ℹ️  INFO",
        }[self.severity]

        result = f"{prefix} [{self.code}] {self.path}: {self.message}"
        if self.suggestion:
            result += f"\n   💡 建议: {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    """验证结果."""

    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    doc: DevDoc | None = None

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)

    def get_summary(self) -> str:
        """获取验证摘要."""
        if self.valid:
            status = "✅ 验证通过"
        else:
            status = "❌ 验证失败"

        return (
            f"{status}\n"
            f"  错误: {self.error_count} | 警告: {self.warning_count} | 信息: {self.info_count}"
        )

    def format_report(self) -> str:
        """格式化完整报告."""
        lines = [
            "=" * 60,
            "开发文档验证报告",
            "=" * 60,
            "",
            self.get_summary(),
            "",
        ]

        if self.issues:
            lines.append("-" * 60)
            for issue in self.issues:
                lines.append(str(issue))
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class DocValidator:
    """开发文档验证器.

    提供全面的文档验证，包括：
    1. 格式验证（Pydantic Schema）
    2. 语义验证（依赖关系、ID 唯一性等）
    3. 质量检查（描述完整性、任务粒度等）
    """

    # 错误代码定义
    ERROR_CODES = {
        # 格式错误 (E0xx)
        "E001": "YAML 解析错误",
        "E002": "Schema 验证失败",
        "E003": "必填字段缺失",
        "E004": "字段类型错误",
        "E005": "字段格式不正确",

        # 语义错误 (E1xx)
        "E101": "模块 ID 重复",
        "E102": "任务 ID 重复",
        "E103": "循环依赖",
        "E104": "依赖模块不存在",
        "E105": "任务 ID 格式不正确",
        "E106": "模块 ID 格式不正确",

        # 警告 (W0xx)
        "W001": "描述过短",
        "W002": "验收标准过于模糊",
        "W003": "任务粒度过大",
        "W004": "任务粒度过小",
        "W005": "文件路径可能不存在",
        "W006": "优先级分配不合理",
        "W007": "缺少测试任务",
        "W008": "TODO 占位符未填写",

        # 信息 (I0xx)
        "I001": "建议添加更多详情",
        "I002": "可以拆分为多个任务",
        "I003": "建议指定 AI 模型",
    }

    def __init__(
        self,
        strict_mode: bool = False,
        check_file_paths: bool = False,
        project_root: Path | None = None,
    ):
        """初始化验证器.

        Args:
            strict_mode: 严格模式，警告也视为错误
            check_file_paths: 是否检查文件路径是否存在
            project_root: 项目根目录（用于检查文件路径）
        """
        self._strict_mode = strict_mode
        self._check_file_paths = check_file_paths
        self._project_root = project_root

    def validate_file(self, path: Path | str) -> ValidationResult:
        """验证开发文档文件.

        Args:
            path: 文件路径

        Returns:
            验证结果
        """
        path = Path(path)
        result = ValidationResult()

        # 检查文件存在
        if not path.exists():
            result.valid = False
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                path=str(path),
                message=f"文件不存在: {path}",
                code="E001",
            ))
            return result

        # 读取并解析
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                else:
                    import json
                    data = json.load(f)
        except yaml.YAMLError as e:
            result.valid = False
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                path=str(path),
                message=f"YAML 解析错误: {e}",
                suggestion="请检查 YAML 语法，确保缩进正确",
                code="E001",
            ))
            return result
        except Exception as e:
            result.valid = False
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                path=str(path),
                message=f"文件读取错误: {e}",
                code="E001",
            ))
            return result

        return self.validate_dict(data)

    def validate_dict(self, data: dict[str, Any]) -> ValidationResult:
        """验证开发文档字典.

        Args:
            data: 文档数据字典

        Returns:
            验证结果
        """
        result = ValidationResult()

        # 1. Schema 验证
        try:
            doc = DevDoc(**data)
            result.doc = doc
        except ValidationError as e:
            result.valid = False
            for error in e.errors():
                loc_path = ".".join(str(l) for l in error["loc"])
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path=loc_path,
                    message=self._format_pydantic_error(error),
                    suggestion=self._get_fix_suggestion(error),
                    code="E002",
                ))
            return result

        # 2. 语义验证
        self._validate_semantics(doc, result)

        # 3. 质量检查
        self._validate_quality(doc, result)

        # 严格模式下，警告也算失败
        if self._strict_mode and result.warning_count > 0:
            result.valid = False

        # 有错误则失败
        if result.error_count > 0:
            result.valid = False

        return result

    def validate_yaml(self, yaml_content: str) -> ValidationResult:
        """验证 YAML 字符串.

        Args:
            yaml_content: YAML 内容

        Returns:
            验证结果
        """
        result = ValidationResult()

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            result.valid = False
            # 解析 YAML 错误位置
            line_info = ""
            if hasattr(e, "problem_mark"):
                mark = e.problem_mark
                line_info = f" (行 {mark.line + 1}, 列 {mark.column + 1})"

            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                path=f"yaml{line_info}",
                message=str(e),
                suggestion="请检查 YAML 语法，常见问题：缩进不一致、冒号后缺少空格、特殊字符未转义",
                code="E001",
            ))
            return result

        return self.validate_dict(data)

    def _validate_semantics(self, doc: DevDoc, result: ValidationResult) -> None:
        """验证语义正确性."""
        module_ids = set()
        task_ids = set()
        module_id_pattern = re.compile(r"^mod-[a-z0-9-]+$")
        task_id_pattern = re.compile(r"^task-\d{3,}$")

        # 检查模块 ID 唯一性和格式
        for i, module in enumerate(doc.modules):
            path = f"modules[{i}]"

            # 模块 ID 格式
            if not module_id_pattern.match(module.id):
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path=f"{path}.id",
                    message=f"模块 ID '{module.id}' 格式不正确",
                    suggestion="模块 ID 应为 'mod-' 开头，后跟小写字母、数字和连字符，如 'mod-user-auth'",
                    code="E106",
                ))

            # 模块 ID 唯一性
            if module.id in module_ids:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    path=f"{path}.id",
                    message=f"模块 ID '{module.id}' 重复",
                    suggestion="每个模块必须有唯一的 ID",
                    code="E101",
                ))
            module_ids.add(module.id)

            # 检查子任务
            for j, task in enumerate(module.sub_tasks):
                task_path = f"{path}.sub_tasks[{j}]"

                # 任务 ID 格式
                if not task_id_pattern.match(task.id):
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        path=f"{task_path}.id",
                        message=f"任务 ID '{task.id}' 格式不正确",
                        suggestion="任务 ID 应为 'task-' 开头，后跟至少3位数字，如 'task-001'",
                        code="E105",
                    ))

                # 任务 ID 唯一性（全局）
                if task.id in task_ids:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        path=f"{task_path}.id",
                        message=f"任务 ID '{task.id}' 重复",
                        suggestion="任务 ID 在整个文档中必须唯一",
                        code="E102",
                    ))
                task_ids.add(task.id)

        # 检查依赖关系
        for i, module in enumerate(doc.modules):
            path = f"modules[{i}]"
            for dep in module.dependencies:
                if dep not in module_ids:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        path=f"{path}.dependencies",
                        message=f"依赖的模块 '{dep}' 不存在",
                        suggestion=f"可用的模块 ID: {', '.join(module_ids)}",
                        code="E104",
                    ))

        # 检查循环依赖
        if self._has_circular_dependency(doc):
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                path="modules",
                message="检测到循环依赖",
                suggestion="请检查模块的 dependencies 配置，确保不存在 A->B->C->A 的循环",
                code="E103",
            ))

    def _validate_quality(self, doc: DevDoc, result: ValidationResult) -> None:
        """验证文档质量."""
        has_test_task = False

        for i, module in enumerate(doc.modules):
            path = f"modules[{i}]"

            # 检查描述长度
            if len(module.description) < 20:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    path=f"{path}.description",
                    message="模块描述过短",
                    suggestion="建议提供更详细的模块描述，至少 20 个字符",
                    code="W001",
                ))

            # 检查验收标准
            if len(module.acceptance_criteria) < 15:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    path=f"{path}.acceptance_criteria",
                    message="验收标准过于简短",
                    suggestion="验收标准应明确、可测试，建议包含具体的功能检查点",
                    code="W002",
                ))

            # 检查 TODO 占位符
            if "TODO" in module.description or "TODO" in module.acceptance_criteria:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    path=path,
                    message="包含未填写的 TODO 占位符",
                    suggestion="请将 TODO 替换为实际内容",
                    code="W008",
                ))

            # 检查子任务
            for j, task in enumerate(module.sub_tasks):
                task_path = f"{path}.sub_tasks[{j}]"

                # 任务粒度检查
                if task.estimated_minutes > 90:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=f"{task_path}.estimated_minutes",
                        message=f"任务预估时间过长 ({task.estimated_minutes} 分钟)",
                        suggestion="建议将大任务拆分为多个小任务，每个不超过 60 分钟",
                        code="W003",
                    ))

                if task.estimated_minutes < 10:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        path=f"{task_path}.estimated_minutes",
                        message=f"任务预估时间较短 ({task.estimated_minutes} 分钟)",
                        suggestion="考虑合并多个小任务",
                        code="W004",
                    ))

                # 需求描述检查
                if len(task.requirements) < 30:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=f"{task_path}.requirements",
                        message="任务需求描述过短",
                        suggestion="详细的需求描述有助于 AI 更好地完成任务",
                        code="W001",
                    ))

                # TODO 检查
                if "TODO" in task.requirements or "TODO" in task.description:
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        path=task_path,
                        message="包含未填写的 TODO 占位符",
                        suggestion="请将 TODO 替换为实际内容",
                        code="W008",
                    ))

                # 检查是否有测试任务
                if task.type.value == "test":
                    has_test_task = True

                # 检查文件路径
                if self._check_file_paths and self._project_root:
                    for file_path in task.files_involved:
                        full_path = self._project_root / file_path
                        if not full_path.exists() and not full_path.parent.exists():
                            result.issues.append(ValidationIssue(
                                severity=ValidationSeverity.INFO,
                                path=f"{task_path}.files_involved",
                                message=f"文件路径可能不存在: {file_path}",
                                suggestion="如果是新文件，可忽略此提示",
                                code="W005",
                            ))

        # 检查是否缺少测试任务
        if not has_test_task:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                path="modules",
                message="文档中没有测试类型的任务",
                suggestion="建议添加测试任务确保代码质量",
                code="W007",
            ))

    def _has_circular_dependency(self, doc: DevDoc) -> bool:
        """检查是否存在循环依赖."""
        # 构建邻接表
        graph: dict[str, list[str]] = {}
        for module in doc.modules:
            graph[module.id] = module.dependencies

        # DFS 检测环
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def _format_pydantic_error(self, error: dict[str, Any]) -> str:
        """格式化 Pydantic 错误消息."""
        error_type = error.get("type", "")
        msg = error.get("msg", "未知错误")

        # 常见错误类型的中文说明
        type_messages = {
            "missing": "必填字段缺失",
            "string_type": "应为字符串类型",
            "int_type": "应为整数类型",
            "list_type": "应为列表类型",
            "enum": "值不在允许范围内",
            "string_pattern_mismatch": "字符串格式不匹配",
            "value_error": "值验证失败",
            "extra_forbidden": "不允许的额外字段",
        }

        for key, message in type_messages.items():
            if key in error_type:
                return f"{message}: {msg}"

        return msg

    def _get_fix_suggestion(self, error: dict[str, Any]) -> str:
        """获取修复建议."""
        error_type = error.get("type", "")
        loc = error.get("loc", [])

        suggestions = {
            "missing": "请添加此必填字段",
            "string_pattern_mismatch": "请检查格式要求，参考示例",
            "enum": "请使用允许的值之一",
            "extra_forbidden": "请删除此字段或检查拼写",
            "int_type": "请使用整数值",
            "list_type": "请使用列表格式 (以 - 开头的项目)",
        }

        for key, suggestion in suggestions.items():
            if key in error_type:
                return suggestion

        # 根据字段位置给出具体建议
        if loc:
            field = str(loc[-1])
            if field == "id":
                return "ID 格式：模块用 'mod-xxx'，任务用 'task-001'"
            if field == "type":
                return "允许的类型：feature, bugfix, refactor, test, docs"

        return "请参考文档格式要求"


def validate_dev_doc_file(path: Path | str, strict: bool = False) -> ValidationResult:
    """便捷函数：验证开发文档文件.

    Args:
        path: 文件路径
        strict: 严格模式

    Returns:
        验证结果
    """
    validator = DocValidator(strict_mode=strict)
    return validator.validate_file(path)


def validate_dev_doc_yaml(yaml_content: str, strict: bool = False) -> ValidationResult:
    """便捷函数：验证 YAML 字符串.

    Args:
        yaml_content: YAML 内容
        strict: 严格模式

    Returns:
        验证结果
    """
    validator = DocValidator(strict_mode=strict)
    return validator.validate_yaml(yaml_content)
