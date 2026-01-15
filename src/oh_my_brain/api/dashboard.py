"""Web Dashboard API.

提供 RESTful API 和 WebSocket 实时更新用于 Web 仪表板。
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# 检查 FastAPI 是否可用
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    WebSocket = None  # type: ignore
    WebSocketDisconnect = Exception  # type: ignore
    CORSMiddleware = None  # type: ignore
    JSONResponse = None  # type: ignore
    BaseModel = object  # type: ignore


# ============================================================
# Pydantic 模型
# ============================================================

if FASTAPI_AVAILABLE:
    from pydantic import BaseModel as PydanticBaseModel

    class TaskCreate(PydanticBaseModel):
        """创建任务请求."""

        module_id: str
        name: str
        description: str
        type: str = "development"
        requirements: list[str] = []
        files_involved: list[str] = []
        priority: int = 0
        depends_on: list[str] = []

    class TaskResponse(PydanticBaseModel):
        """任务响应."""

        id: str
        module_id: str
        name: str
        type: str
        status: str
        priority: int
        created_at: str | None = None
        updated_at: str | None = None
        assigned_worker: str | None = None

    class WorkerResponse(PydanticBaseModel):
        """Worker 响应."""

        worker_id: str
        hostname: str
        platform: str
        status: str
        current_task_id: str | None = None
        last_heartbeat: str

    class ApprovalRequest(PydanticBaseModel):
        """审批请求."""

        request_id: str
        approved: bool
        comment: str = ""

    class StatusResponse(PydanticBaseModel):
        """状态响应."""

        running: bool
        workers_count: int
        tasks_pending: int
        tasks_running: int
        tasks_completed: int
        uptime_seconds: float

else:
    # 当 FastAPI 不可用时的占位符
    class TaskCreate:
        pass

    class TaskResponse:
        pass

    class WorkerResponse:
        pass

    class ApprovalRequest:
        pass

    class StatusResponse:
        pass


# ============================================================
# WebSocket 连接管理
# ============================================================


class ConnectionManager:
    """WebSocket 连接管理器."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """接受 WebSocket 连接."""
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.info(f"WebSocket connected, total: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """断开 WebSocket 连接."""
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        logger.info(f"WebSocket disconnected, total: {len(self._connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """广播消息给所有连接."""
        if not self._connections:
            return

        data = json.dumps(message)
        disconnected = []

        async with self._lock:
            for conn in self._connections:
                try:
                    await conn.send_text(data)
                except Exception:
                    disconnected.append(conn)

            # 清理断开的连接
            for conn in disconnected:
                if conn in self._connections:
                    self._connections.remove(conn)


# 全局连接管理器
ws_manager = ConnectionManager()


# ============================================================
# Dashboard API 应用
# ============================================================


def create_dashboard_app(brain_server: Any = None) -> FastAPI | None:
    """创建 Dashboard FastAPI 应用.

    Args:
        brain_server: BrainServer 实例，用于获取状态和执行操作

    Returns:
        FastAPI 应用实例，如果 FastAPI 不可用则返回 None
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, dashboard disabled. Install with: pip install 'oh-my-brain[dashboard]'")
        return None

    app = FastAPI(
        title="Oh My Brain Dashboard API",
        description="Multi-Agent 协作开发框架管理接口",
        version="0.1.0",
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 存储 Brain 引用
    app.state.brain = brain_server
    app.state.start_time = datetime.now()
    app.state.pending_approvals: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------
    # 健康检查端点
    # --------------------------------------------------------

    @app.get("/api/health")
    async def health_check():
        """健康检查."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        """获取系统状态."""
        if app.state.brain:
            status = app.state.brain.get_status()
            uptime = (datetime.now() - app.state.start_time).total_seconds()
            tasks = status.get("tasks", {})
            return StatusResponse(
                running=status.get("running", False),
                workers_count=len(status.get("workers", {})),
                tasks_pending=tasks.get("pending", 0),
                tasks_running=tasks.get("running", 0),
                tasks_completed=tasks.get("completed", 0),
                uptime_seconds=uptime,
            )
        return StatusResponse(
            running=False,
            workers_count=0,
            tasks_pending=0,
            tasks_running=0,
            tasks_completed=0,
            uptime_seconds=0,
        )

    # --------------------------------------------------------
    # 任务管理端点
    # --------------------------------------------------------

    @app.get("/api/tasks")
    async def list_tasks():
        """获取任务列表."""
        if not app.state.brain:
            return []

        status = app.state.brain.get_status()
        tasks = status.get("tasks", {}).get("all_tasks", [])
        return tasks

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        """获取单个任务详情."""
        if not app.state.brain:
            raise HTTPException(status_code=503, detail="Brain not available")

        status = app.state.brain.get_status()
        tasks = status.get("tasks", {}).get("all_tasks", [])
        for task in tasks:
            if task.get("id") == task_id:
                return task
        raise HTTPException(status_code=404, detail="Task not found")

    @app.post("/api/tasks")
    async def create_task(task: TaskCreate):
        """创建新任务."""
        if not app.state.brain:
            raise HTTPException(status_code=503, detail="Brain not available")

        try:
            # 通过 Brain 的调度器创建任务
            scheduler = app.state.brain._task_scheduler
            new_task = scheduler.add_task(
                module_id=task.module_id,
                name=task.name,
                description=task.description,
                task_type=task.type,
                requirements=task.requirements,
                files_involved=task.files_involved,
                priority=task.priority,
                depends_on=task.depends_on,
            )

            # 广播任务创建事件
            await ws_manager.broadcast({
                "event": "task_created",
                "data": {"task_id": new_task.id, "name": new_task.name},
            })

            return {"id": new_task.id, "status": "created"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/tasks/{task_id}")
    async def cancel_task(task_id: str):
        """取消任务."""
        if not app.state.brain:
            raise HTTPException(status_code=503, detail="Brain not available")

        try:
            scheduler = app.state.brain._task_scheduler
            scheduler.mark_cancelled(task_id)

            await ws_manager.broadcast({
                "event": "task_cancelled",
                "data": {"task_id": task_id},
            })

            return {"task_id": task_id, "status": "cancelled"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # --------------------------------------------------------
    # Worker 管理端点
    # --------------------------------------------------------

    @app.get("/api/workers")
    async def list_workers():
        """获取 Worker 列表."""
        if not app.state.brain:
            return []

        status = app.state.brain.get_status()
        workers = status.get("workers", {})
        return list(workers.values())

    @app.get("/api/workers/{worker_id}")
    async def get_worker(worker_id: str):
        """获取单个 Worker 详情."""
        if not app.state.brain:
            raise HTTPException(status_code=503, detail="Brain not available")

        status = app.state.brain.get_status()
        workers = status.get("workers", {})
        if worker_id in workers:
            return workers[worker_id]
        raise HTTPException(status_code=404, detail="Worker not found")

    # --------------------------------------------------------
    # 审批管理端点
    # --------------------------------------------------------

    @app.get("/api/approvals")
    async def list_pending_approvals():
        """获取待审批列表."""
        return list(app.state.pending_approvals.values())

    @app.post("/api/approvals")
    async def add_approval_request(request: dict[str, Any]):
        """添加审批请求（内部使用）."""
        request_id = request.get("id")
        if request_id:
            app.state.pending_approvals[request_id] = request

            await ws_manager.broadcast({
                "event": "approval_required",
                "data": request,
            })

        return {"status": "pending", "request_id": request_id}

    @app.post("/api/approvals/{request_id}/approve")
    async def approve_request(request_id: str, data: ApprovalRequest):
        """批准审批请求."""
        if request_id not in app.state.pending_approvals:
            raise HTTPException(status_code=404, detail="Approval request not found")

        request = app.state.pending_approvals.pop(request_id)
        request["status"] = "approved"
        request["approved_at"] = datetime.now().isoformat()
        request["comment"] = data.comment

        await ws_manager.broadcast({
            "event": "approval_completed",
            "data": {"request_id": request_id, "approved": True},
        })

        return {"request_id": request_id, "status": "approved"}

    @app.post("/api/approvals/{request_id}/reject")
    async def reject_request(request_id: str, data: ApprovalRequest):
        """拒绝审批请求."""
        if request_id not in app.state.pending_approvals:
            raise HTTPException(status_code=404, detail="Approval request not found")

        request = app.state.pending_approvals.pop(request_id)
        request["status"] = "rejected"
        request["rejected_at"] = datetime.now().isoformat()
        request["comment"] = data.comment

        await ws_manager.broadcast({
            "event": "approval_completed",
            "data": {"request_id": request_id, "approved": False},
        })

        return {"request_id": request_id, "status": "rejected"}

    # --------------------------------------------------------
    # WebSocket 端点
    # --------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket 实时更新端点."""
        await ws_manager.connect(websocket)
        try:
            while True:
                # 保持连接，接收心跳
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await ws_manager.disconnect(websocket)

    # --------------------------------------------------------
    # Metrics 端点
    # --------------------------------------------------------

    @app.get("/api/metrics")
    async def get_metrics():
        """获取 Prometheus 格式的指标."""
        from oh_my_brain.brain.metrics import get_metrics_endpoint

        return JSONResponse(
            content=get_metrics_endpoint(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


async def broadcast_event(event: str, data: dict[str, Any]) -> None:
    """广播事件到所有 WebSocket 连接.

    这个函数供其他模块调用，用于推送实时更新。
    """
    await ws_manager.broadcast({"event": event, "data": data})


def run_dashboard_server(
    brain_server: Any = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """运行 Dashboard 服务器.

    Args:
        brain_server: BrainServer 实例
        host: 监听地址
        port: 监听端口
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available. Install with: pip install 'oh-my-brain[dashboard]'")
        return

    try:
        import uvicorn
    except ImportError:
        logger.error("Uvicorn not available. Install with: pip install uvicorn")
        return

    app = create_dashboard_app(brain_server)
    if app:
        uvicorn.run(app, host=host, port=port)
