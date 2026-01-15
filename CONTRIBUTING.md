# Contributing to OH MY BRAIN

感谢你对 OH MY BRAIN 项目的关注！我们欢迎各种形式的贡献。

## 🚀 快速开始

### 开发环境设置

1. **Fork 并克隆仓库**

```bash
git clone https://github.com/YOUR_USERNAME/oh-my-brain.git
cd oh-my-brain
```

2. **安装 uv（推荐的包管理器）**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **安装依赖**

```bash
uv sync --all-groups
```

4. **安装 pre-commit 钩子**

```bash
uv run pre-commit install
```

5. **验证环境**

```bash
uv run pytest
uv run ruff check .
uv run mypy src/
```

## 📝 贡献流程

### 1. 创建 Issue

在开始工作之前，请先创建或认领一个 Issue：

- **Bug 报告**：使用 Bug Report 模板
- **功能请求**：使用 Feature Request 模板
- **问题讨论**：直接创建 Issue 讨论

### 2. 创建分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

分支命名规范：
- `feature/xxx` - 新功能
- `fix/xxx` - Bug 修复
- `docs/xxx` - 文档更新
- `refactor/xxx` - 重构
- `test/xxx` - 测试相关

### 3. 编写代码

请遵循以下规范：

#### 代码风格

- 使用 **ruff** 进行代码格式化和检查
- 使用 **mypy** 进行类型检查
- 所有公共 API 需要类型注解
- 所有公共函数/类需要 docstring

```python
def process_task(task_id: str, config: TaskConfig) -> TaskResult:
    """处理单个任务。

    Args:
        task_id: 任务唯一标识符
        config: 任务配置

    Returns:
        任务执行结果

    Raises:
        TaskNotFoundError: 当任务不存在时
    """
    ...
```

#### 提交信息

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
feat: add new task scheduler
fix: resolve memory leak in context manager
docs: update installation guide
refactor: simplify worker communication protocol
test: add integration tests for brain server
chore: update dependencies
```

### 4. 编写测试

- 所有新功能需要对应的单元测试
- 测试覆盖率目标：80%+
- 使用 pytest 编写测试

```python
# tests/unit/test_context_manager.py
import pytest
from oh_my_brain.brain.context_manager import ContextManager

class TestContextManager:
    @pytest.fixture
    def manager(self):
        return ContextManager(redis_url="redis://localhost")

    async def test_store_context(self, manager):
        await manager.store("worker-1", {"messages": []})
        result = await manager.retrieve("worker-1")
        assert result == {"messages": []}
```

### 5. 提交 Pull Request

1. 确保所有测试通过
2. 确保代码检查通过
3. 更新相关文档
4. 填写 PR 模板

## 🏗 项目结构

```
oh-my-brain/
├── src/oh_my_brain/       # 主代码
│   ├── brain/             # Brain 核心模块
│   ├── worker/            # Worker 模块
│   ├── protocol/          # 通信协议
│   ├── schemas/           # 数据模型
│   ├── platform/          # 平台适配
│   └── dashboard/         # Dashboard
├── tests/                 # 测试
│   ├── unit/              # 单元测试
│   └── integration/       # 集成测试
├── docs/                  # 文档
├── examples/              # 示例
└── config/                # 配置示例
```

## 🔧 开发命令

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/unit/test_brain.py

# 运行测试并生成覆盖率报告
uv run pytest --cov=oh_my_brain --cov-report=html

# 代码格式化
uv run ruff format .

# 代码检查
uv run ruff check .

# 自动修复可修复的问题
uv run ruff check --fix .

# 类型检查
uv run mypy src/

# 启动文档服务
uv run mkdocs serve
```

## 📋 代码审查标准

PR 将根据以下标准审查：

- [ ] 代码风格符合项目规范
- [ ] 类型注解完整
- [ ] 测试覆盖充分
- [ ] 文档已更新（如需要）
- [ ] Commit 信息规范
- [ ] 无破坏性变更（或已标注）

## 🎯 优先级领域

当前我们特别欢迎以下领域的贡献：

1. **Brain 核心功能** - 任务调度、上下文管理
2. **Worker 适配器** - 支持更多 Agent 框架
3. **平台兼容性** - Windows/macOS 测试和优化
4. **文档和示例** - 使用指南、最佳实践
5. **Dashboard** - 监控界面优化

## ❓ 获取帮助

- 📖 查看 [文档](docs/)
- 💬 在 Issue 中提问
- 🔍 搜索已有 Issue

再次感谢你的贡献！🙏
