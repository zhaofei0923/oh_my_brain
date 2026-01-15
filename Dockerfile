# OH MY BRAIN Docker配置
# 多阶段构建，优化镜像大小

FROM python:3.11-slim AS builder

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN pip install --no-cache-dir uv

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY pyproject.toml README.md ./
COPY src ./src

# 创建虚拟环境并安装依赖
RUN uv venv /app/.venv
RUN uv pip install --python /app/.venv/bin/python -e ".[all]"


# ============================================================
# 生产镜像
# ============================================================
FROM python:3.11-slim AS runtime

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd --create-home --shell /bin/bash brain

# 设置工作目录
WORKDIR /app

# 从builder复制虚拟环境
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# 设置环境变量
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# 切换到非root用户
USER brain

# 默认命令
CMD ["oh-my-brain", "--help"]


# ============================================================
# Brain服务镜像
# ============================================================
FROM runtime AS brain

# Brain默认端口
EXPOSE 5555
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import zmq; ctx = zmq.Context(); sock = ctx.socket(zmq.REQ); sock.connect('tcp://localhost:5555'); sock.close()" || exit 1

# 启动Brain服务
CMD ["oh-my-brain", "brain", "start", "--host", "0.0.0.0"]


# ============================================================
# Worker服务镜像
# ============================================================
FROM runtime AS worker

# Worker环境变量
ENV BRAIN_ADDRESS="tcp://brain:5555"

# 启动Worker
CMD ["oh-my-brain", "worker", "start", "--brain", "${BRAIN_ADDRESS}"]
