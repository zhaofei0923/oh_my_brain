"""配置相关数据模型."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrainConfig(BaseSettings):
    """Brain服务配置."""

    model_config = SettingsConfigDict(
        env_prefix="OMB_BRAIN_",
        env_file=".env",
        extra="ignore",
    )

    # 服务配置
    host: str = Field("127.0.0.1", description="监听地址")
    port: int = Field(15555, description="监听端口")
    transport: str = Field("auto", description="传输协议: auto, ipc, tcp")

    # Redis配置
    redis_url: str = Field("redis://localhost:6379/0", description="Redis连接URL")
    redis_prefix: str = Field("omb:", description="Redis键前缀")

    # 上下文管理
    context_token_limit: int = Field(90000, description="上下文token上限")
    context_summary_threshold: float = Field(0.8, description="触发摘要的阈值比例")

    # 任务调度
    max_workers: int = Field(10, description="最大Worker数量")
    task_timeout_minutes: int = Field(60, description="任务超时时间（分钟）")
    heartbeat_interval_seconds: int = Field(5, description="心跳间隔（秒）")
    heartbeat_timeout_seconds: int = Field(30, description="心跳超时（秒）")

    # Git配置
    git_base_branch: str = Field("develop", description="基础分支")
    git_branch_prefix: str = Field("agent/", description="Worker分支前缀")
    git_auto_merge: bool = Field(False, description="是否自动合并（需人工审核）")

    # 安全配置
    safety_check_enabled: bool = Field(True, description="是否启用安全检查")
    dangerous_commands: list[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf ~",
            "chmod 777",
            "curl | bash",
            "wget | bash",
            "> /dev/sda",
            "mkfs",
            "dd if=",
        ],
        description="危险命令模式列表",
    )

    # 日志配置
    log_level: str = Field("INFO", description="日志级别")
    log_file: Path | None = Field(None, description="日志文件路径")

    # 模型配置文件路径
    models_config_path: Path = Field(
        Path("~/.oh-my-brain/models.yaml").expanduser(),
        description="模型配置文件路径",
    )


class WorkerConfig(BaseSettings):
    """Worker配置."""

    model_config = SettingsConfigDict(
        env_prefix="OMB_WORKER_",
        env_file=".env",
        extra="ignore",
    )

    # Worker标识
    worker_id: str | None = Field(None, description="Worker ID，None则自动生成")

    # Brain连接
    brain_host: str = Field("127.0.0.1", description="Brain地址")
    brain_port: int = Field(15555, description="Brain端口")
    brain_transport: str = Field("auto", description="传输协议: auto, ipc, tcp")

    # 心跳
    heartbeat_interval_seconds: int = Field(5, description="心跳间隔（秒）")

    # 并发
    max_concurrent_tasks: int = Field(1, description="最大并发任务数")

    # 工作目录
    workspace_dir: Path = Field(Path("./workspace"), description="工作目录")

    # 资源限制
    max_memory_mb: int = Field(2048, description="最大内存（MB）")
    max_cpu_percent: float = Field(80.0, description="最大CPU使用率")

    # 日志
    log_level: str = Field("INFO", description="日志级别")

    @property
    def brain_address(self) -> str:
        """获取Brain地址."""
        if self.brain_transport == "ipc":
            return "ipc:///tmp/oh-my-brain.sock"
        return f"tcp://{self.brain_host}:{self.brain_port}"


class DashboardConfig(BaseSettings):
    """Dashboard配置."""

    model_config = SettingsConfigDict(
        env_prefix="OMB_DASHBOARD_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field("0.0.0.0", description="监听地址")
    port: int = Field(8080, description="监听端口")
    brain_url: str = Field("tcp://127.0.0.1:15555", description="Brain连接地址")
    redis_url: str = Field("redis://localhost:6379/0", description="Redis连接URL")
