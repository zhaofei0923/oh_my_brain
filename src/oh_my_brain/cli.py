"""OH MY BRAIN 命令行接口."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="oh-my-brain",
    help="Multi-Agent Collaborative Development Framework",
    add_completion=False,
)

brain_app = typer.Typer(help="Brain server commands")
worker_app = typer.Typer(help="Worker commands")
dev_doc_app = typer.Typer(help="Development document commands")

app.add_typer(brain_app, name="brain")
app.add_typer(worker_app, name="worker")
app.add_typer(dev_doc_app, name="doc")

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """配置日志."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


# ============================================================
# Brain 命令
# ============================================================


@brain_app.command("start")
def brain_start(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind",
    ),
    port: int = typer.Option(
        5555,
        "--port",
        "-p",
        help="Port to bind",
    ),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Number of workers to spawn (0 = none)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
) -> None:
    """启动Brain服务器."""
    setup_logging(verbose)

    console.print(
        Panel.fit(
            "[bold blue]OH MY BRAIN[/bold blue]\n[dim]Multi-Agent Collaborative Development[/dim]",
            border_style="blue",
        )
    )

    console.print(f"Starting Brain server on {host}:{port}...")

    from oh_my_brain.brain.server import BrainServer
    from oh_my_brain.schemas.config import BrainConfig

    # 加载配置
    if config and config.exists():
        import yaml

        with open(config) as f:
            config_data = yaml.safe_load(f)
        brain_config = BrainConfig.model_validate(config_data)
    else:
        brain_config = BrainConfig(
            host=host,
            port=port,
        )

    # 创建并启动服务器
    server = BrainServer(brain_config)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Brain server stopped[/yellow]")


@brain_app.command("status")
def brain_status(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(5555, "--port", "-p"),
) -> None:
    """查看Brain状态."""
    console.print(f"Checking Brain status at {host}:{port}...")

    import zmq

    address = f"tcp://{host}:{port}"

    try:
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3秒超时
        socket.setsockopt(zmq.SNDTIMEO, 3000)
        socket.connect(address)

        # 发送心跳检测
        socket.send_json({"type": "ping"})

        try:
            response = socket.recv_json()
            console.print(f"[green]✓ Brain is running at {address}[/green]")
            if response:
                console.print(f"  Response: {response}")
        except zmq.Again:
            console.print(f"[yellow]⚠ Brain server at {address} is not responding[/yellow]")
            console.print("  The server may be busy or not running.")

        socket.close()
        context.term()

    except zmq.ZMQError as e:
        console.print(f"[red]✗ Cannot connect to Brain at {address}[/red]")
        console.print(f"  Error: {e}")
        console.print("\n[yellow]To start the Brain server:[/yellow]")
        console.print("  oh-my-brain brain start")


# ============================================================
# Worker 命令
# ============================================================


@worker_app.command("start")
def worker_start(
    brain_address: str = typer.Option(
        "tcp://127.0.0.1:5555",
        "--brain",
        "-b",
        help="Brain server address",
    ),
    worker_id: str | None = typer.Option(
        None,
        "--id",
        help="Worker ID (auto-generated if not provided)",
    ),
    capabilities: str | None = typer.Option(
        None,
        "--caps",
        help="Comma-separated capabilities",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
) -> None:
    """启动Worker."""
    setup_logging(verbose)

    caps_list = capabilities.split(",") if capabilities else []

    console.print(f"Starting Worker, connecting to {brain_address}...")

    from oh_my_brain.schemas.config import WorkerConfig
    from oh_my_brain.worker.mini_agent_adapter import MiniAgentAdapter

    config = WorkerConfig(
        brain_address=brain_address,
        worker_id=worker_id,
    )

    worker = MiniAgentAdapter(config)
    worker.set_capabilities(caps_list)

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Worker stopped[/yellow]")


@worker_app.command("list")
def worker_list(
    brain_address: str = typer.Option(
        "tcp://127.0.0.1:5555",
        "--brain",
        "-b",
    ),
) -> None:
    """列出所有Worker."""
    console.print("Fetching worker list...")

    import zmq

    try:
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.RCVTIMEO, 3000)
        socket.setsockopt(zmq.SNDTIMEO, 3000)
        socket.connect(brain_address)

        # 请求 Worker 列表
        socket.send_json({"type": "list_workers"})

        try:
            response = socket.recv_json()

            table = Table(title="Workers")
            table.add_column("ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Current Task")
            table.add_column("Capabilities")

            workers = response.get("workers", [])
            if workers:
                for w in workers:
                    status = "active" if w.get("active") else "idle"
                    task = w.get("current_task", "-")
                    caps = ", ".join(w.get("capabilities", [])) or "-"
                    table.add_row(w.get("id", "unknown"), status, task, caps)
                console.print(table)
            else:
                console.print("[yellow]No workers connected[/yellow]")

        except zmq.Again:
            console.print("[yellow]Brain server is not responding[/yellow]")

        socket.close()
        context.term()

    except zmq.ZMQError as e:
        console.print(f"[red]Cannot connect to Brain: {e}[/red]")
        console.print("\n[yellow]Make sure the Brain server is running:[/yellow]")
        console.print("  oh-my-brain brain start")


# ============================================================
# Dev Doc 命令
# ============================================================


@dev_doc_app.command("validate")
def doc_validate(
    file: Path = typer.Argument(..., help="YAML file to validate"),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Strict mode: warnings are also errors",
    ),
    output_format: str = typer.Option(
        "full",
        "--format",
        "-f",
        help="Output format: full, summary, json",
    ),
) -> None:
    """验证开发文档（增强版）."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    from oh_my_brain.doc.validator import DocValidator

    validator = DocValidator(strict_mode=strict)
    result = validator.validate_file(file)

    if output_format == "json":
        import json
        output = {
            "valid": result.valid,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
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
        console.print(json.dumps(output, ensure_ascii=False, indent=2))
    elif output_format == "summary":
        console.print(result.get_summary())
    else:
        console.print(result.format_report())

    if not result.valid:
        raise typer.Exit(1)


@dev_doc_app.command("template")
def doc_template(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path",
    ),
) -> None:
    """生成开发文档模板."""
    from oh_my_brain.brain.doc_parser import DocParser

    template = DocParser.generate_template()

    if output:
        output.write_text(template, encoding="utf-8")
        console.print(f"[green]Template saved to: {output}[/green]")
    else:
        console.print(template)


@dev_doc_app.command("schema")
def doc_schema(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path",
    ),
) -> None:
    """导出JSON Schema."""
    import json

    from oh_my_brain.brain.doc_parser import DocParser

    schema = DocParser.export_json_schema()
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)

    if output:
        output.write_text(schema_json, encoding="utf-8")
        console.print(f"[green]Schema saved to: {output}[/green]")
    else:
        console.print(schema_json)


@dev_doc_app.command("generate")
def doc_generate(
    project_name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Project name",
    ),
    requirements: Path | None = typer.Option(
        None,
        "--requirements",
        "-r",
        help="Requirements file path",
    ),
    requirements_text: str | None = typer.Option(
        None,
        "--text",
        "-t",
        help="Requirements text (direct input)",
    ),
    project_type: str = typer.Option(
        "web_api",
        "--type",
        "-p",
        help="Project type: web_api, web_frontend, h5_mobile, saas_platform, data_platform, cpp_algorithm, etc.",
    ),
    output: Path = typer.Option(
        Path("dev_doc.yaml"),
        "--output",
        "-o",
        help="Output file path",
    ),
    use_llm: bool = typer.Option(
        True,
        "--use-llm/--no-llm",
        help="Use LLM for generation (requires API key)",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="MINIMAX_API_KEY",
        help="MiniMax API key",
    ),
) -> None:
    """使用 LLM 生成开发文档（增强版）."""
    from oh_my_brain.doc.generator import DocGenerator, GenerationMode, ProjectType as PT, save_dev_doc

    # 获取需求文本
    req_text = ""
    if requirements and requirements.exists():
        req_text = requirements.read_text(encoding="utf-8")
    elif requirements_text:
        req_text = requirements_text
    elif use_llm:
        console.print("[red]需要提供 --requirements 或 --text 参数[/red]")
        raise typer.Exit(1)

    # 解析项目类型
    try:
        pt = PT(project_type)
    except ValueError:
        console.print(f"[red]未知项目类型: {project_type}[/red]")
        console.print(f"可用类型: {', '.join([t.value for t in PT])}")
        raise typer.Exit(1)

    console.print(f"[bold blue]📝 正在生成开发文档: {project_name}[/bold blue]")
    console.print(f"   项目类型: {project_type}")
    console.print(f"   使用 LLM: {use_llm}")
    console.print()

    mode = GenerationMode.AUTO if use_llm else GenerationMode.MANUAL
    generator = DocGenerator(api_key=api_key, project_type=pt, mode=mode)

    try:
        if use_llm and api_key:
            doc = asyncio.run(
                generator.generate_from_requirements(
                    project_name=project_name,
                    requirements=req_text,
                )
            )
            console.print("[green]✨ LLM 生成完成[/green]")
        else:
            doc = generator.create_from_template(project_name)
            console.print("[green]✨ 模板生成完成[/green]")

        save_dev_doc(doc, output)
        console.print(f"[green]📁 文档已保存: {output}[/green]")

        # 显示摘要
        console.print()
        console.print("[bold]📊 文档摘要:[/bold]")
        console.print(f"   模块数: {len(doc.modules)}")
        total_tasks = sum(len(m.sub_tasks) for m in doc.modules)
        console.print(f"   任务数: {total_tasks}")

    except Exception as e:
        console.print(f"[red]❌ 生成失败: {e}[/red]")
        raise typer.Exit(1)


@dev_doc_app.command("show")
def doc_show(
    file: Path = typer.Argument(..., help="Development document file"),
    output_format: str = typer.Option(
        "tree",
        "--format",
        "-f",
        help="Output format: tree, table, json",
    ),
) -> None:
    """显示开发文档内容."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    from oh_my_brain.doc.updater import DocUpdater

    updater = DocUpdater()
    try:
        updater.load_from_file(file)
    except Exception as e:
        console.print(f"[red]无法加载文档: {e}[/red]")
        raise typer.Exit(1)

    doc = updater.doc
    if not doc:
        console.print("[red]文档为空[/red]")
        raise typer.Exit(1)

    if output_format == "json":
        import json
        console.print(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2))
    elif output_format == "table":
        _show_doc_table(doc)
    else:
        _show_doc_tree(doc)


def _show_doc_tree(doc) -> None:
    """树形显示文档."""
    console.print(f"[bold blue]📦 {doc.project_name}[/bold blue]")
    console.print(f"├── 版本: {doc.version}")
    console.print(f"├── 描述: {doc.description}")
    console.print(f"└── 模块 ({len(doc.modules)}):")

    for i, module in enumerate(doc.modules):
        is_last = (i == len(doc.modules) - 1)
        prefix = "    └──" if is_last else "    ├──"
        child_prefix = "       " if is_last else "    │  "

        console.print(f"{prefix} [yellow]📁 {module.name}[/yellow] ({module.id})")
        console.print(f"{child_prefix} ├── 优先级: P{module.priority}")
        console.print(f"{child_prefix} ├── 依赖: {', '.join(module.dependencies) or '无'}")
        console.print(f"{child_prefix} └── 任务 ({len(module.sub_tasks)}):")

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

            desc = task.description[:35] + "..." if len(task.description) > 35 else task.description
            console.print(f"{task_prefix} {type_emoji} [dim]{task.id}[/dim]: {desc}")


def _show_doc_table(doc) -> None:
    """表格显示文档."""
    console.print()
    console.print(f"[bold blue]项目: {doc.project_name}[/bold blue]")
    console.print()

    for module in doc.modules:
        table = Table(title=f"[{module.id}] {module.name} (P{module.priority})")
        table.add_column("ID", style="dim")
        table.add_column("类型", width=10)
        table.add_column("描述", width=40)
        table.add_column("时间", justify="right")

        for task in module.sub_tasks:
            desc = task.description[:38] + "..." if len(task.description) > 38 else task.description
            table.add_row(
                task.id,
                task.type.value,
                desc,
                f"{task.estimated_minutes}m",
            )

        console.print(table)
        console.print()


@dev_doc_app.command("add-module")
def doc_add_module(
    file: Path = typer.Argument(..., help="Development document file"),
    module_id: str = typer.Option(..., "--id", help="Module ID (e.g., mod-user-auth)"),
    name: str = typer.Option(..., "--name", "-n", help="Module name"),
    description: str = typer.Option(..., "--description", "-d", help="Module description"),
    priority: int = typer.Option(2, "--priority", "-p", help="Priority (1-3)"),
) -> None:
    """添加模块到开发文档."""
    from oh_my_brain.doc.updater import DocUpdater
    from oh_my_brain.schemas.dev_doc import Module

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
        updater.add_module(module)
        updater.commit(f"添加模块: {module_id}")
        updater.save(file)
        console.print(f"[green]✅ 已添加模块: {module_id}[/green]")
    except ValueError as e:
        console.print(f"[red]❌ 添加失败: {e}[/red]")
        raise typer.Exit(1)


@dev_doc_app.command("add-task")
def doc_add_task(
    file: Path = typer.Argument(..., help="Development document file"),
    module_id: str = typer.Option(..., "--module", "-m", help="Target module ID"),
    description: str = typer.Option(..., "--description", "-d", help="Task description"),
    requirements: str = typer.Option(..., "--requirements", "-r", help="Task requirements"),
    task_type: str = typer.Option("feature", "--type", "-t", help="Task type: feature, bugfix, refactor, test, docs"),
    minutes: int = typer.Option(30, "--minutes", help="Estimated minutes"),
) -> None:
    """添加任务到模块."""
    from oh_my_brain.doc.updater import DocUpdater
    from oh_my_brain.schemas.dev_doc import SubTask, TaskType

    updater = DocUpdater()
    updater.load_from_file(file)

    task_id = updater.generate_next_task_id()

    try:
        tt = TaskType(task_type)
    except ValueError:
        console.print(f"[red]未知任务类型: {task_type}[/red]")
        raise typer.Exit(1)

    task = SubTask(
        id=task_id,
        description=description,
        type=tt,
        requirements=requirements,
        files_involved=[],
        estimated_minutes=minutes,
    )

    try:
        updater.add_task(module_id, task)
        updater.commit(f"添加任务: {task_id}")
        updater.save(file)
        console.print(f"[green]✅ 已添加任务: {task_id} -> {module_id}[/green]")
    except ValueError as e:
        console.print(f"[red]❌ 添加失败: {e}[/red]")
        raise typer.Exit(1)


@dev_doc_app.command("types")
def doc_types() -> None:
    """列出支持的项目类型和模板."""
    from oh_my_brain.doc.generator import PROJECT_TEMPLATES, ProjectType

    console.print("[bold blue]📋 支持的项目类型:[/bold blue]\n")

    for pt in ProjectType:
        template = PROJECT_TEMPLATES.get(pt, {})
        tech_stack = template.get("tech_stack", [])
        modules = template.get("common_modules", [])

        console.print(f"  [yellow]{pt.value}[/yellow]")
        if tech_stack:
            console.print(f"  ├── 技术栈: {', '.join(tech_stack[:5])}")
        if modules:
            console.print(f"  └── 常用模块: {', '.join(modules[:5])}")
        console.print()


@dev_doc_app.command("run")
def doc_run(
    file: Path = typer.Argument(..., help="Development document to run"),
    brain_address: str = typer.Option(
        "tcp://127.0.0.1:5555",
        "--brain",
        "-b",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show execution plan without running",
    ),
) -> None:
    """执行开发文档."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    from oh_my_brain.brain.doc_parser import DocParser

    # 解析文档
    try:
        dev_doc = DocParser.parse_file(file)
    except ValueError as e:
        console.print(f"[red]Failed to parse document: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Loaded project: {dev_doc.project.name}[/green]")
    console.print(f"Modules: {len(dev_doc.modules)}")

    total_tasks = sum(len(m.sub_tasks) for m in dev_doc.modules)
    console.print(f"Total tasks: {total_tasks}")

    if dry_run:
        # 显示执行计划
        table = Table(title="Execution Plan")
        table.add_column("Order", style="dim")
        table.add_column("Module")
        table.add_column("Task")
        table.add_column("Type")
        table.add_column("Est. Time")

        order = 1
        for module in dev_doc.modules:
            for task in module.sub_tasks:
                table.add_row(
                    str(order),
                    module.name,
                    task.name,
                    task.type,
                    f"{task.estimated_minutes}m",
                )
                order += 1

        console.print(table)
        return

    # 实际执行
    asyncio.run(_execute_dev_doc(dev_doc, brain_address, console))


async def _execute_dev_doc(dev_doc, brain_address: str, console: Console) -> None:
    """执行开发文档中的任务.

    Args:
        dev_doc: 解析后的开发文档
        brain_address: Brain服务器地址
        console: Rich控制台
    """
    from oh_my_brain.brain.task_scheduler import TaskScheduler
    from oh_my_brain.schemas.task import TaskStatus

    console.print("\n[bold blue]Starting execution...[/bold blue]\n")

    # 创建任务调度器
    scheduler = TaskScheduler()
    scheduler.load_from_dev_doc(dev_doc)

    # 显示任务统计
    all_tasks = scheduler.get_all_tasks()
    pending_count = len([t for t in all_tasks if t.status == TaskStatus.PENDING])
    console.print(f"Loaded {len(all_tasks)} tasks, {pending_count} pending")

    # 检查是否有可用的 Worker
    console.print("\n[yellow]Note: Make sure Brain server and Workers are running:[/yellow]")
    console.print("  1. oh-my-brain brain start")
    console.print("  2. oh-my-brain worker start")
    console.print("")

    # 连接到 Brain 并提交任务
    import zmq
    import zmq.asyncio

    try:
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.DEALER)
        socket.connect(brain_address)

        console.print(f"Connected to Brain at {brain_address}")

        # 这里可以实现任务提交逻辑
        # 但更好的方式是让 Brain 自动从 DevDoc 加载任务
        console.print("\n[green]Tasks are ready for execution.[/green]")
        console.print("Workers will automatically pick up tasks from the Brain.")

        # 显示任务列表
        task_table = Table(title="Pending Tasks")
        task_table.add_column("ID", style="dim")
        task_table.add_column("Name")
        task_table.add_column("Type")
        task_table.add_column("Dependencies")

        for task in all_tasks:
            if task.status == TaskStatus.PENDING:
                deps = ", ".join(task.depends_on) if task.depends_on else "-"
                task_table.add_row(
                    task.id[:8],
                    task.name,
                    task.task_type.value if task.task_type else "unknown",
                    deps,
                )

        console.print(task_table)

        socket.close()
        context.term()

    except Exception as e:
        console.print(f"[red]Failed to connect to Brain: {e}[/red]")
        console.print("[yellow]Make sure the Brain server is running.[/yellow]")


# ============================================================
# 主命令
# ============================================================


@app.command("version")
def version() -> None:
    """显示版本信息."""
    from oh_my_brain import __version__

    console.print(f"oh-my-brain version {__version__}")


@app.command("init")
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Project path",
    ),
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help="Project template (basic, fastapi, flask)",
    ),
) -> None:
    """初始化新项目."""
    console.print(f"Initializing project at {path}...")

    # 创建目录结构
    (path / "config").mkdir(parents=True, exist_ok=True)
    (path / "docs").mkdir(parents=True, exist_ok=True)

    # 生成配置文件
    from oh_my_brain.brain.doc_parser import DocParser

    template_content = DocParser.generate_template()
    (path / "dev_doc.yaml").write_text(template_content, encoding="utf-8")

    console.print("[green]Project initialized![/green]")
    console.print("Next steps:")
    console.print("  1. Edit dev_doc.yaml with your project requirements")
    console.print("  2. Run: oh-my-brain brain start")
    console.print("  3. Run: oh-my-brain doc run dev_doc.yaml")


def main() -> None:
    """主入口."""
    app()


if __name__ == "__main__":
    main()
