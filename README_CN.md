# 🧠 OH MY BRAIN

[![CI](https://github.com/YOUR_USERNAME/oh-my-brain/workflows/CI/badge.svg)](https://github.com/YOUR_USERNAME/oh-my-brain/actions)
[![PyPI](https://img.shields.io/pypi/v/oh-my-brain)](https://pypi.org/project/oh-my-brain/)
[![Python](https://img.shields.io/pypi/pyversions/oh-my-brain)](https://pypi.org/project/oh-my-brain/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[English](README.md) | [中文](README_CN.md)

> 多Agent协作开发框架 - 让AI团队为你并行编程

## ✨ 特性

- 🧠 **中央大脑协调** - Brain统一管理任务分配、上下文、Git操作
- 🤖 **多Worker并行** - 多个AI Agent同时开发不同模块
- 🔌 **灵活AI配置** - 按任务类型选择不同模型（费用/性能自主权衡）
- 📋 **标准化开发文档** - YAML格式，可用任意AI生成
- 🔒 **安全审核** - 危险命令预审，保护你的代码库
- 📊 **实时监控** - Dashboard查看每个Worker的进展
- 🖥️ **跨平台支持** - 支持Linux、macOS、Windows和WSL

## 🏗 架构

```
                    ┌─────────────────────────────────┐
                    │            BRAIN                │
                    │  ┌─────────┐ ┌─────────┐       │
                    │  │ 上下文   │ │  模型   │       │
                    │  │ 管理器   │ │  路由   │       │
                    │  └────┬────┘ └────┬────┘       │
                    │  ┌────┴──────────┴────┐       │
                    │  │     任务调度器      │       │
                    │  └─────────┬──────────┘       │
                    └────────────┼─────────────────┘
                                 │ ZeroMQ
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
      ┌────────────┐      ┌────────────┐      ┌────────────┐
      │  Worker 1  │      │  Worker 2  │      │  Worker N  │
      │(Mini-Agent)│      │(Mini-Agent)│      │(Mini-Agent)│
      └────────────┘      └────────────┘      └────────────┘
```

## 🚀 快速开始

### 安装

```bash
pip install oh-my-brain
```

或使用 uv（推荐）：

```bash
uv add oh-my-brain
```

### 1. 配置AI模型

```yaml
# ~/.oh-my-brain/models.yaml
models:
  - name: "deepseek-coder"
    provider: "openai"
    api_base: "https://api.deepseek.com"
    model: "deepseek-coder"
    api_key_env: "DEEPSEEK_API_KEY"
    cost_per_1k_tokens: 0.001
    capabilities: [code]

  - name: "minimax-m21"
    provider: "anthropic"
    api_base: "https://api.minimax.io"
    model: "MiniMax-M2.1"
    api_key_env: "MINIMAX_API_KEY"
    cost_per_1k_tokens: 0.002
    capabilities: [code, reasoning, planning]

task_model_mapping:
  planning: "minimax-m21"      # 规划任务用MiniMax
  coding: "deepseek-coder"      # 编码任务用DeepSeek（便宜）
  review: "minimax-m21"         # 审查任务用MiniMax
  default: "deepseek-coder"     # 默认模型
```

### 2. 准备开发文档

用你喜欢的AI（Claude、DeepSeek、GPT等）生成标准化开发文档：

```yaml
# my_project/dev_doc.yaml
project:
  name: "my-awesome-app"
  version: "0.1.0"
  description: "一个示例Web应用"
  tech_stack:
    language: "Python"
    framework: "FastAPI"
    database: "PostgreSQL"

modules:
  - id: "mod-auth"
    name: "用户认证模块"
    description: "用户认证和授权功能"
    priority: 1
    dependencies: []
    acceptance_criteria: "用户可以注册、登录和登出"
    sub_tasks:
      - id: "task-001"
        name: "实现JWT登录"
        type: "feature"
        description: "创建登录接口，生成JWT令牌"
        estimated_minutes: 30
        files_involved:
          - "src/auth/router.py"
          - "src/auth/service.py"
        requirements: |
          - POST /auth/login 接口
          - 接受邮箱和密码
          - 成功后返回JWT令牌
```

### 3. 启动Brain

```bash
oh-my-brain start --config ./brain.yaml
```

### 4. 启动Workers

```bash
oh-my-brain worker --count 4  # 启动4个Worker
```

### 5. 提交开发文档

```bash
oh-my-brain submit ./dev_doc.yaml
```

然后在Dashboard中观看AI团队为你工作！

## 📖 文档

- [快速开始](docs/getting-started/)
- [开发文档格式](docs/guides/dev-doc-format.md)
- [模型配置指南](docs/guides/model-config.md)
- [API参考](docs/api/)

## 🖥️ 平台支持

| 平台 | 支持状态 | 说明 |
|------|----------|------|
| Linux | ✅ 完全支持 | 推荐生产环境 |
| macOS | ✅ 完全支持 | |
| Windows | ✅ 支持 | 使用TCP传输，性能略低 |
| WSL | ✅ 完全支持 | Windows用户推荐 |

### Windows用户建议

1. **开发/测试**：原生Windows完全可用
2. **生产部署**：建议使用WSL或Linux服务器
3. **多Worker场景**：建议4-6个Worker（Windows进程开销较大）

## 🛠 开发

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/oh-my-brain.git
cd oh-my-brain

# 安装依赖
uv sync --all-groups

# 运行测试
uv run pytest

# 运行代码检查
uv run ruff check .
uv run mypy src/
```

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 许可证

Apache 2.0 - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- [Mini-Agent](https://github.com/MiniMax-AI/Mini-Agent) - Worker agent的基础框架
- [MiniMax](https://www.minimax.io/) - AI模型提供商
