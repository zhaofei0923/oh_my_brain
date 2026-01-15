# Todo API 示例项目

这是一个使用 OH MY BRAIN 开发的示例项目，展示如何使用开发文档驱动 Multi-Agent 协作开发。

## 项目描述

一个简单的 Todo REST API，使用：
- FastAPI 框架
- SQLite 数据库
- SQLAlchemy ORM
- Pydantic 数据验证

## 使用 OH MY BRAIN 构建

### 1. 确保 Brain 和 Worker 正在运行

```bash
# 终端1: 启动Brain
oh-my-brain brain start

# 终端2: 启动Worker
oh-my-brain worker start
```

### 2. 验证开发文档

```bash
oh-my-brain doc validate examples/todo_api/dev_doc.yaml
```

### 3. 预览执行计划

```bash
oh-my-brain doc run examples/todo_api/dev_doc.yaml --dry-run
```

### 4. 执行开发

```bash
oh-my-brain doc run examples/todo_api/dev_doc.yaml
```

## 文档结构

`dev_doc.yaml` 定义了以下模块和任务：

1. **项目初始化** - 创建项目结构
2. **数据模型** - SQLAlchemy 模型定义
3. **Pydantic Schemas** - 请求/响应验证
4. **CRUD操作** - 数据库操作函数
5. **API路由** - REST 端点
6. **单元测试** - pytest 测试

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/todos | 获取所有 Todo |
| POST | /api/todos | 创建 Todo |
| GET | /api/todos/{id} | 获取单个 Todo |
| PUT | /api/todos/{id} | 更新 Todo |
| DELETE | /api/todos/{id} | 删除 Todo |

## 运行项目

完成开发后：

```bash
cd examples/todo_api
pip install -r requirements.txt
uvicorn src.main:app --reload
```

访问 http://localhost:8000/docs 查看 Swagger 文档。
