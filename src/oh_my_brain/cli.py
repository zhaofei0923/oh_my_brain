"""OH MY BRAIN 命令行接口."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

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
    config: Optional[Path] = typer.Option(
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
            "[bold blue]OH MY BRAIN[/bold blue]\n"
            "[dim]Multi-Agent Collaborative Development[/dim]",
            border_style="blue",
        )
    )

    console.print(f"Starting Brain server on {host}:{port}...")

    from oh_my_brain.schemas.config import BrainConfig
    from oh_my_brain.brain.server import BrainServer

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

    # TODO: 实现状态检查
    console.print("[yellow]Status check not yet implemented[/yellow]")


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
    worker_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Worker ID (auto-generated if not provided)",
    ),
    capabilities: Optional[str] = typer.Option(
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

    # TODO: 实现worker列表获取
    table = Table(title="Workers")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Current Task")
    table.add_column("Capabilities")

    console.print(table)
    console.print("[yellow]Worker list not yet implemented[/yellow]")


# ============================================================
# Dev Doc 命令
# ============================================================


@dev_doc_app.command("validate")
def doc_validate(
    file: Path = typer.Argument(..., help="YAML file to validate"),
) -> None:
    """验证开发文档."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    from oh_my_brain.brain.doc_parser import DocParser

    valid, error = DocParser.validate_file(file)

    if valid:
        console.print(f"[green]✓ Valid development document: {file}[/green]")
    else:
        console.print(f"[red]✗ Invalid document: {error}[/red]")
        raise typer.Exit(1)


@dev_doc_app.command("template")
def doc_template(
    output: Optional[Path] = typer.Option(
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
    output: Optional[Path] = typer.Option(
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
    console.print("[yellow]Execution not yet implemented[/yellow]")


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
