"""开发文档 CLI 命令.

提供文档生成、验证、更新的命令行接口。
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

from oh_my_brain.doc import (
    DocGenerator,
    DocUpdater,
    DocValidator,
    GenerationMode,
    ProjectType,
    save_dev_doc,
    validate_dev_doc_file,
)
from oh_my_brain.schemas.dev_doc import Module, SubTask, TaskType


@click.group("doc")
def doc_cli() -> None:
    """开发文档管理命令."""
    pass


# ========== 生成命令 ==========

@doc_cli.command("generate")
@click.option(
    "--requirements", "-r",
    type=click.Path(exists=True),
    help="需求文件路径",
)
@click.option(
    "--requirements-text", "-t",
    type=str,
    help="需求文本（直接输入）",
)
@click.option(
    "--project-type", "-p",
    type=click.Choice([t.value for t in ProjectType]),
    default=ProjectType.WEB_API.value,
    help="项目类型",
)
@click.option(
    "--project-name", "-n",
    type=str,
    required=True,
    help="项目名称",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="dev_doc.yaml",
    help="输出文件路径",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["auto", "interactive", "manual"]),
    default="auto",
    help="生成模式",
)
@click.option(
    "--api-key",
    type=str,
    envvar="MINIMAX_API_KEY",
    help="MiniMax API Key",
)
def generate_doc(
    requirements: Optional[str],
    requirements_text: Optional[str],
    project_type: str,
    project_name: str,
    output: str,
    mode: str,
    api_key: Optional[str],
) -> None:
    """使用 LLM 生成开发文档.

    示例:
        # 从需求文件生成
        brain doc generate -r requirements.txt -n my_project -o dev_doc.yaml

        # 从文本生成
        brain doc generate -t "开发一个用户管理系统" -n user_system

        # 使用模板（不调用 LLM）
        brain doc generate -n my_project -p web_api -m manual
    """
    # 获取需求文本
    req_text = ""
    if requirements:
        with open(requirements, "r", encoding="utf-8") as f:
            req_text = f.read()
    elif requirements_text:
        req_text = requirements_text
    elif mode != "manual":
        click.echo("错误: 需要提供 --requirements 或 --requirements-text", err=True)
        sys.exit(1)

    # 解析模式
    gen_mode = {
        "auto": GenerationMode.AUTO,
        "interactive": GenerationMode.INTERACTIVE,
        "manual": GenerationMode.MANUAL,
    }[mode]

    # 创建生成器
    generator = DocGenerator(
        api_key=api_key,
        project_type=ProjectType(project_type),
        mode=gen_mode,
    )

    click.echo(f"📝 正在生成开发文档: {project_name}")
    click.echo(f"   项目类型: {project_type}")
    click.echo(f"   生成模式: {mode}")
    click.echo()

    try:
        if mode == "manual":
            # 使用模板生成
            doc = generator.create_from_template(project_name)
            click.echo("✨ 使用模板生成完成")
        else:
            # 使用 LLM 生成
            if not api_key:
                click.echo("警告: 未提供 API Key，将使用模板模式", err=True)
                doc = generator.create_from_template(project_name)
            else:
                doc = asyncio.run(
                    generator.generate_from_requirements(
                        project_name=project_name,
                        requirements=req_text,
                    )
                )
                click.echo("✨ LLM 生成完成")

        # 保存文档
        output_path = Path(output)
        save_dev_doc(doc, output_path)
        click.echo(f"📁 文档已保存: {output_path}")

        # 显示摘要
        click.echo()
        click.echo("📊 文档摘要:")
        click.echo(f"   模块数: {len(doc.modules)}")
        total_tasks = sum(len(m.sub_tasks) for m in doc.modules)
        click.echo(f"   任务数: {total_tasks}")

    except Exception as e:
        click.echo(f"❌ 生成失败: {e}", err=True)
        sys.exit(1)


# ========== 验证命令 ==========

@doc_cli.command("validate")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--strict", "-s",
    is_flag=True,
    help="严格模式（警告也视为错误）",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["full", "summary", "json"]),
    default="full",
    help="输出格式",
)
@click.option(
    "--check-paths",
    is_flag=True,
    help="检查文件路径是否存在",
)
@click.option(
    "--project-root",
    type=click.Path(exists=True),
    help="项目根目录（用于检查路径）",
)
def validate_doc(
    file: str,
    strict: bool,
    format: str,
    check_paths: bool,
    project_root: Optional[str],
) -> None:
    """验证开发文档格式.

    示例:
        # 基本验证
        brain doc validate dev_doc.yaml

        # 严格模式
        brain doc validate dev_doc.yaml --strict

        # 检查文件路径
        brain doc validate dev_doc.yaml --check-paths --project-root ./
    """
    validator = DocValidator(
        strict_mode=strict,
        check_file_paths=check_paths,
        project_root=Path(project_root) if project_root else None,
    )

    result = validator.validate_file(file)

    if format == "json":
        import json
        output = {
            "valid": result.valid,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "info_count": result.info_count,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "code": issue.code,
                    "path": issue.path,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
                for issue in result.issues
            ],
        }
        click.echo(json.dumps(output, ensure_ascii=False, indent=2))
    elif format == "summary":
        click.echo(result.get_summary())
    else:
        click.echo(result.format_report())

    # 设置退出码
    if not result.valid:
        sys.exit(1)


# ========== 更新命令 ==========

@doc_cli.group("update")
def update_group() -> None:
    """文档更新命令组."""
    pass


@update_group.command("add-module")
@click.argument("file", type=click.Path(exists=True))
@click.option("--id", "module_id", required=True, help="模块 ID")
@click.option("--name", required=True, help="模块名称")
@click.option("--description", "-d", required=True, help="模块描述")
@click.option("--priority", "-p", type=int, default=2, help="优先级")
@click.option("--reason", "-r", default="", help="添加原因")
def add_module(
    file: str,
    module_id: str,
    name: str,
    description: str,
    priority: int,
    reason: str,
) -> None:
    """添加模块到开发文档.

    示例:
        brain doc update add-module dev_doc.yaml \\
            --id mod-new-feature \\
            --name "新功能模块" \\
            --description "实现 XX 新功能"
    """
    updater = DocUpdater()
    updater.load_from_file(file)

    module = Module(
        id=module_id,
        name=name,
        description=description,
        priority=priority,
        acceptance_criteria="TODO: 填写验收标准",
        sub_tasks=[],
        dependencies=[],
    )

    try:
        updater.add_module(module, reason=reason)
        updater.commit(f"添加模块: {module_id}")
        updater.save(file)
        click.echo(f"✅ 已添加模块: {module_id}")
    except ValueError as e:
        click.echo(f"❌ 添加失败: {e}", err=True)
        sys.exit(1)


@update_group.command("remove-module")
@click.argument("file", type=click.Path(exists=True))
@click.option("--id", "module_id", required=True, help="模块 ID")
@click.option("--reason", "-r", default="", help="删除原因")
@click.option("--force", "-f", is_flag=True, help="强制删除")
def remove_module(
    file: str,
    module_id: str,
    reason: str,
    force: bool,
) -> None:
    """从开发文档删除模块.

    示例:
        brain doc update remove-module dev_doc.yaml --id mod-old-feature
    """
    updater = DocUpdater()
    updater.load_from_file(file)

    module = updater.get_module(module_id)
    if not module:
        click.echo(f"❌ 模块不存在: {module_id}", err=True)
        sys.exit(1)

    if not force and module.sub_tasks:
        click.echo(f"⚠️  模块 {module_id} 包含 {len(module.sub_tasks)} 个任务")
        if not click.confirm("确定要删除吗?"):
            click.echo("已取消")
            return

    try:
        updater.remove_module(module_id, reason=reason)
        updater.commit(f"删除模块: {module_id}")
        updater.save(file)
        click.echo(f"✅ 已删除模块: {module_id}")
    except ValueError as e:
        click.echo(f"❌ 删除失败: {e}", err=True)
        sys.exit(1)


@update_group.command("add-task")
@click.argument("file", type=click.Path(exists=True))
@click.option("--module", "-m", required=True, help="目标模块 ID")
@click.option("--id", "task_id", help="任务 ID（自动生成）")
@click.option("--description", "-d", required=True, help="任务描述")
@click.option("--requirements", "-r", required=True, help="任务需求")
@click.option(
    "--type", "task_type",
    type=click.Choice(["feature", "bugfix", "refactor", "test", "docs"]),
    default="feature",
    help="任务类型",
)
@click.option("--minutes", type=int, default=30, help="预估分钟数")
@click.option("--files", multiple=True, help="涉及文件")
def add_task(
    file: str,
    module: str,
    task_id: Optional[str],
    description: str,
    requirements: str,
    task_type: str,
    minutes: int,
    files: tuple,
) -> None:
    """添加任务到模块.

    示例:
        brain doc update add-task dev_doc.yaml \\
            --module mod-user \\
            --description "实现用户注册" \\
            --requirements "支持邮箱和手机号注册" \\
            --files src/user.py --files tests/test_user.py
    """
    updater = DocUpdater()
    updater.load_from_file(file)

    # 自动生成任务 ID
    if not task_id:
        task_id = updater.generate_next_task_id()

    task = SubTask(
        id=task_id,
        description=description,
        type=TaskType(task_type),
        requirements=requirements,
        files_involved=list(files),
        estimated_minutes=minutes,
    )

    try:
        updater.add_task(module, task)
        updater.commit(f"添加任务: {task_id}")
        updater.save(file)
        click.echo(f"✅ 已添加任务: {task_id} -> {module}")
    except ValueError as e:
        click.echo(f"❌ 添加失败: {e}", err=True)
        sys.exit(1)


@update_group.command("remove-task")
@click.argument("file", type=click.Path(exists=True))
@click.option("--id", "task_id", required=True, help="任务 ID")
@click.option("--reason", "-r", default="", help="删除原因")
def remove_task(file: str, task_id: str, reason: str) -> None:
    """从开发文档删除任务.

    示例:
        brain doc update remove-task dev_doc.yaml --id task-005
    """
    updater = DocUpdater()
    updater.load_from_file(file)

    try:
        updater.remove_task(task_id, reason=reason)
        updater.commit(f"删除任务: {task_id}")
        updater.save(file)
        click.echo(f"✅ 已删除任务: {task_id}")
    except ValueError as e:
        click.echo(f"❌ 删除失败: {e}", err=True)
        sys.exit(1)


# ========== 查看命令 ==========

@doc_cli.command("show")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--format", "-f",
    type=click.Choice(["tree", "table", "json"]),
    default="tree",
    help="输出格式",
)
def show_doc(file: str, format: str) -> None:
    """显示开发文档内容.

    示例:
        brain doc show dev_doc.yaml
        brain doc show dev_doc.yaml --format table
    """
    updater = DocUpdater()
    updater.load_from_file(file)

    doc = updater.doc
    if not doc:
        click.echo("❌ 无法加载文档", err=True)
        sys.exit(1)

    if format == "json":
        import json
        click.echo(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2))
    elif format == "table":
        _show_table(doc)
    else:
        _show_tree(doc)


def _show_tree(doc) -> None:
    """树形显示."""
    click.echo(f"📦 {doc.project_name}")
    click.echo(f"├── 版本: {doc.version}")
    click.echo(f"├── 描述: {doc.description}")
    click.echo(f"└── 模块 ({len(doc.modules)}):")

    for i, module in enumerate(doc.modules):
        is_last_module = (i == len(doc.modules) - 1)
        prefix = "    └──" if is_last_module else "    ├──"
        child_prefix = "       " if is_last_module else "    │  "

        click.echo(f"{prefix} 📁 {module.name} ({module.id})")
        click.echo(f"{child_prefix} ├── 优先级: P{module.priority}")
        click.echo(f"{child_prefix} ├── 依赖: {', '.join(module.dependencies) or '无'}")
        click.echo(f"{child_prefix} └── 任务 ({len(module.sub_tasks)}):")

        for j, task in enumerate(module.sub_tasks):
            is_last_task = (j == len(module.sub_tasks) - 1)
            task_prefix = f"{child_prefix}     └──" if is_last_task else f"{child_prefix}     ├──"

            type_emoji = {
                "feature": "✨",
                "bugfix": "🐛",
                "refactor": "♻️",
                "test": "🧪",
                "docs": "📝",
            }.get(task.type.value, "📋")

            click.echo(f"{task_prefix} {type_emoji} {task.id}: {task.description[:40]}")


def _show_table(doc) -> None:
    """表格显示."""
    click.echo(f"\n{'='*70}")
    click.echo(f"项目: {doc.project_name}")
    click.echo(f"{'='*70}\n")

    for module in doc.modules:
        click.echo(f"[{module.id}] {module.name} (P{module.priority})")
        click.echo(f"{'─'*50}")
        click.echo(f"{'ID':<12} {'类型':<10} {'描述':<30} {'时间':<8}")
        click.echo(f"{'─'*50}")

        for task in module.sub_tasks:
            desc = task.description[:28] + "..." if len(task.description) > 28 else task.description
            click.echo(f"{task.id:<12} {task.type.value:<10} {desc:<30} {task.estimated_minutes}m")

        click.echo()


# ========== 历史命令 ==========

@doc_cli.command("history")
@click.argument("file", type=click.Path(exists=True))
@click.option("--limit", "-n", type=int, default=10, help="显示条数")
def show_history(file: str, limit: int) -> None:
    """显示文档变更历史.

    示例:
        brain doc history dev_doc.yaml
    """
    # 需要历史目录
    file_path = Path(file)
    history_dir = file_path.parent / ".doc_history"

    updater = DocUpdater(history_dir=history_dir)
    updater.load_from_file(file)

    history = updater.get_version_history()

    if not history:
        click.echo("暂无变更历史")
        return

    click.echo(f"📜 变更历史 (共 {len(history)} 个版本)")
    click.echo(f"{'─'*50}")

    for version in history[-limit:]:
        ts = version["timestamp"][:19].replace("T", " ")
        click.echo(f"v{version['version']} | {ts}")
        for change in version.get("changes", []):
            click.echo(f"  └── {change['description']}")


# ========== 模板命令 ==========

@doc_cli.command("templates")
def list_templates() -> None:
    """列出可用的项目模板."""
    from oh_my_brain.doc.generator import PROJECT_TEMPLATES

    click.echo("📋 可用项目模板:\n")

    for project_type in ProjectType:
        template = PROJECT_TEMPLATES.get(project_type, {})
        tech_stack = template.get("tech_stack", [])
        modules = template.get("common_modules", [])

        click.echo(f"  {project_type.value}")
        click.echo(f"  ├── 技术栈: {', '.join(tech_stack[:5])}")
        click.echo(f"  └── 常用模块: {', '.join(modules[:5])}")
        click.echo()


# ========== 导出模块 ==========

def register_doc_commands(cli: click.Group) -> None:
    """注册文档命令到主 CLI."""
    cli.add_command(doc_cli)
