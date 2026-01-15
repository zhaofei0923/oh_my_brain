"""开发文档更新器.

支持运行时动态修改开发文档，包括版本管理和变更追踪。
"""

import json
import logging
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from oh_my_brain.schemas.dev_doc import DevDoc, Module, SubTask, TaskType

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """变更类型."""

    ADD_MODULE = "add_module"
    REMOVE_MODULE = "remove_module"
    UPDATE_MODULE = "update_module"
    ADD_TASK = "add_task"
    REMOVE_TASK = "remove_task"
    UPDATE_TASK = "update_task"
    REORDER_MODULES = "reorder_modules"
    REORDER_TASKS = "reorder_tasks"
    UPDATE_PROJECT_INFO = "update_project_info"


@dataclass
class ChangeRecord:
    """变更记录."""

    change_type: ChangeType
    timestamp: datetime
    description: str
    path: str  # 变更位置
    old_value: Any = None
    new_value: Any = None
    reason: str = ""  # 变更原因

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
        }


@dataclass
class DocVersion:
    """文档版本."""

    version: str
    timestamp: datetime
    doc_hash: str
    changes: list[ChangeRecord] = field(default_factory=list)
    doc_snapshot: dict[str, Any] | None = None  # 可选的完整快照

    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "doc_hash": self.doc_hash,
            "changes": [c.to_dict() for c in self.changes],
        }


class DocUpdater:
    """开发文档更新器.

    功能：
    1. 动态添加/删除/修改模块和任务
    2. 版本管理和变更追踪
    3. 保存和加载历史版本
    4. 支持回滚到指定版本
    """

    def __init__(
        self,
        doc: DevDoc | None = None,
        history_dir: Path | None = None,
        max_versions: int = 50,
    ):
        """初始化更新器.

        Args:
            doc: 初始文档
            history_dir: 历史版本保存目录
            max_versions: 最大保存版本数
        """
        self._doc = doc
        self._history_dir = history_dir
        self._max_versions = max_versions
        self._versions: list[DocVersion] = []
        self._current_version = "0.0.0"
        self._pending_changes: list[ChangeRecord] = []

        if doc:
            self._create_initial_version()

    @property
    def doc(self) -> DevDoc | None:
        """获取当前文档."""
        return self._doc

    @property
    def current_version(self) -> str:
        """获取当前版本."""
        return self._current_version

    @property
    def version_count(self) -> int:
        """获取版本数量."""
        return len(self._versions)

    def load_doc(self, doc: DevDoc) -> None:
        """加载文档.

        Args:
            doc: 开发文档
        """
        self._doc = doc
        self._create_initial_version()
        self._pending_changes.clear()

    def load_from_file(self, path: Path | str) -> None:
        """从文件加载文档.

        Args:
            path: 文件路径
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        self._doc = DevDoc(**data)
        self._create_initial_version()
        self._pending_changes.clear()

        # 尝试加载历史版本
        if self._history_dir:
            self._load_version_history()

    def _create_initial_version(self) -> None:
        """创建初始版本."""
        if not self._doc:
            return

        doc_hash = self._compute_hash()
        version = DocVersion(
            version="1.0.0",
            timestamp=datetime.now(),
            doc_hash=doc_hash,
            changes=[],
            doc_snapshot=self._doc.model_dump(),
        )
        self._versions = [version]
        self._current_version = "1.0.0"

    def _compute_hash(self) -> str:
        """计算文档哈希."""
        if not self._doc:
            return ""
        doc_json = json.dumps(self._doc.model_dump(), sort_keys=True)
        return hashlib.sha256(doc_json.encode()).hexdigest()[:16]

    def _increment_version(self, change_type: ChangeType) -> str:
        """递增版本号.

        Args:
            change_type: 变更类型

        Returns:
            新版本号
        """
        major, minor, patch = map(int, self._current_version.split("."))

        # 根据变更类型决定版本增量
        if change_type in [ChangeType.ADD_MODULE, ChangeType.REMOVE_MODULE]:
            minor += 1
            patch = 0
        else:
            patch += 1

        return f"{major}.{minor}.{patch}"

    # ========== 模块操作 ==========

    def add_module(
        self,
        module: Module,
        position: int | None = None,
        reason: str = "",
    ) -> None:
        """添加模块.

        Args:
            module: 模块
            position: 插入位置（None 表示末尾）
            reason: 变更原因
        """
        if not self._doc:
            raise ValueError("未加载文档")

        # 检查 ID 唯一性
        existing_ids = {m.id for m in self._doc.modules}
        if module.id in existing_ids:
            raise ValueError(f"模块 ID '{module.id}' 已存在")

        if position is None:
            self._doc.modules.append(module)
            position = len(self._doc.modules) - 1
        else:
            self._doc.modules.insert(position, module)

        change = ChangeRecord(
            change_type=ChangeType.ADD_MODULE,
            timestamp=datetime.now(),
            description=f"添加模块: {module.id} ({module.name})",
            path=f"modules[{position}]",
            new_value=module.model_dump(),
            reason=reason,
        )
        self._pending_changes.append(change)
        logger.info(f"添加模块: {module.id}")

    def remove_module(self, module_id: str, reason: str = "") -> Module:
        """删除模块.

        Args:
            module_id: 模块 ID
            reason: 变更原因

        Returns:
            被删除的模块
        """
        if not self._doc:
            raise ValueError("未加载文档")

        for i, module in enumerate(self._doc.modules):
            if module.id == module_id:
                removed = self._doc.modules.pop(i)

                # 移除其他模块对此模块的依赖
                for m in self._doc.modules:
                    if module_id in m.dependencies:
                        m.dependencies.remove(module_id)

                change = ChangeRecord(
                    change_type=ChangeType.REMOVE_MODULE,
                    timestamp=datetime.now(),
                    description=f"删除模块: {module_id}",
                    path=f"modules[{i}]",
                    old_value=removed.model_dump(),
                    reason=reason,
                )
                self._pending_changes.append(change)
                logger.info(f"删除模块: {module_id}")
                return removed

        raise ValueError(f"模块 '{module_id}' 不存在")

    def update_module(
        self,
        module_id: str,
        updates: dict[str, Any],
        reason: str = "",
    ) -> None:
        """更新模块.

        Args:
            module_id: 模块 ID
            updates: 更新内容
            reason: 变更原因
        """
        if not self._doc:
            raise ValueError("未加载文档")

        for i, module in enumerate(self._doc.modules):
            if module.id == module_id:
                old_value = module.model_dump()

                # 应用更新
                for key, value in updates.items():
                    if key == "sub_tasks":
                        continue  # 子任务单独处理
                    if hasattr(module, key):
                        setattr(module, key, value)

                change = ChangeRecord(
                    change_type=ChangeType.UPDATE_MODULE,
                    timestamp=datetime.now(),
                    description=f"更新模块: {module_id}",
                    path=f"modules[{i}]",
                    old_value=old_value,
                    new_value=module.model_dump(),
                    reason=reason,
                )
                self._pending_changes.append(change)
                logger.info(f"更新模块: {module_id}")
                return

        raise ValueError(f"模块 '{module_id}' 不存在")

    def get_module(self, module_id: str) -> Module | None:
        """获取模块.

        Args:
            module_id: 模块 ID

        Returns:
            模块（未找到返回 None）
        """
        if not self._doc:
            return None

        for module in self._doc.modules:
            if module.id == module_id:
                return module
        return None

    # ========== 任务操作 ==========

    def add_task(
        self,
        module_id: str,
        task: SubTask,
        position: int | None = None,
        reason: str = "",
    ) -> None:
        """添加任务.

        Args:
            module_id: 模块 ID
            task: 任务
            position: 插入位置（None 表示末尾）
            reason: 变更原因
        """
        if not self._doc:
            raise ValueError("未加载文档")

        module = self.get_module(module_id)
        if not module:
            raise ValueError(f"模块 '{module_id}' 不存在")

        # 检查任务 ID 唯一性（全局）
        all_task_ids = set()
        for m in self._doc.modules:
            for t in m.sub_tasks:
                all_task_ids.add(t.id)

        if task.id in all_task_ids:
            raise ValueError(f"任务 ID '{task.id}' 已存在")

        if position is None:
            module.sub_tasks.append(task)
            position = len(module.sub_tasks) - 1
        else:
            module.sub_tasks.insert(position, task)

        change = ChangeRecord(
            change_type=ChangeType.ADD_TASK,
            timestamp=datetime.now(),
            description=f"添加任务: {task.id} 到模块 {module_id}",
            path=f"modules[{module_id}].sub_tasks[{position}]",
            new_value=task.model_dump(),
            reason=reason,
        )
        self._pending_changes.append(change)
        logger.info(f"添加任务: {task.id} -> {module_id}")

    def remove_task(self, task_id: str, reason: str = "") -> SubTask:
        """删除任务.

        Args:
            task_id: 任务 ID
            reason: 变更原因

        Returns:
            被删除的任务
        """
        if not self._doc:
            raise ValueError("未加载文档")

        for module in self._doc.modules:
            for i, task in enumerate(module.sub_tasks):
                if task.id == task_id:
                    removed = module.sub_tasks.pop(i)

                    change = ChangeRecord(
                        change_type=ChangeType.REMOVE_TASK,
                        timestamp=datetime.now(),
                        description=f"删除任务: {task_id}",
                        path=f"modules[{module.id}].sub_tasks[{i}]",
                        old_value=removed.model_dump(),
                        reason=reason,
                    )
                    self._pending_changes.append(change)
                    logger.info(f"删除任务: {task_id}")
                    return removed

        raise ValueError(f"任务 '{task_id}' 不存在")

    def update_task(
        self,
        task_id: str,
        updates: dict[str, Any],
        reason: str = "",
    ) -> None:
        """更新任务.

        Args:
            task_id: 任务 ID
            updates: 更新内容
            reason: 变更原因
        """
        if not self._doc:
            raise ValueError("未加载文档")

        for module in self._doc.modules:
            for i, task in enumerate(module.sub_tasks):
                if task.id == task_id:
                    old_value = task.model_dump()

                    # 应用更新
                    for key, value in updates.items():
                        if hasattr(task, key):
                            if key == "type" and isinstance(value, str):
                                value = TaskType(value)
                            setattr(task, key, value)

                    change = ChangeRecord(
                        change_type=ChangeType.UPDATE_TASK,
                        timestamp=datetime.now(),
                        description=f"更新任务: {task_id}",
                        path=f"modules[{module.id}].sub_tasks[{i}]",
                        old_value=old_value,
                        new_value=task.model_dump(),
                        reason=reason,
                    )
                    self._pending_changes.append(change)
                    logger.info(f"更新任务: {task_id}")
                    return

        raise ValueError(f"任务 '{task_id}' 不存在")

    def get_task(self, task_id: str) -> tuple[Module, SubTask] | None:
        """获取任务及其所属模块.

        Args:
            task_id: 任务 ID

        Returns:
            (模块, 任务) 元组，未找到返回 None
        """
        if not self._doc:
            return None

        for module in self._doc.modules:
            for task in module.sub_tasks:
                if task.id == task_id:
                    return (module, task)
        return None

    # ========== 版本管理 ==========

    def commit(self, message: str = "") -> DocVersion:
        """提交变更，创建新版本.

        Args:
            message: 提交消息

        Returns:
            新版本
        """
        if not self._doc:
            raise ValueError("未加载文档")

        if not self._pending_changes:
            logger.warning("没有待提交的变更")
            return self._versions[-1] if self._versions else None

        # 确定版本增量
        primary_change = self._pending_changes[0].change_type
        new_version = self._increment_version(primary_change)

        version = DocVersion(
            version=new_version,
            timestamp=datetime.now(),
            doc_hash=self._compute_hash(),
            changes=list(self._pending_changes),
            doc_snapshot=self._doc.model_dump(),
        )

        self._versions.append(version)
        self._current_version = new_version
        self._pending_changes.clear()

        # 保存到文件
        if self._history_dir:
            self._save_version(version)

        # 清理旧版本
        while len(self._versions) > self._max_versions:
            removed = self._versions.pop(0)
            if self._history_dir:
                self._remove_version_file(removed)

        logger.info(f"提交版本: {new_version}, 变更数: {len(version.changes)}")
        return version

    def rollback(self, version: str) -> None:
        """回滚到指定版本.

        Args:
            version: 目标版本号
        """
        for v in self._versions:
            if v.version == version:
                if v.doc_snapshot:
                    self._doc = DevDoc(**v.doc_snapshot)
                    self._current_version = version
                    self._pending_changes.clear()
                    logger.info(f"回滚到版本: {version}")
                    return
                else:
                    raise ValueError(f"版本 {version} 没有完整快照，无法回滚")

        raise ValueError(f"版本 '{version}' 不存在")

    def get_version_history(self) -> list[dict[str, Any]]:
        """获取版本历史.

        Returns:
            版本历史列表
        """
        return [v.to_dict() for v in self._versions]

    def get_changes_between(
        self,
        from_version: str,
        to_version: str,
    ) -> list[ChangeRecord]:
        """获取两个版本之间的变更.

        Args:
            from_version: 起始版本
            to_version: 目标版本

        Returns:
            变更记录列表
        """
        changes = []
        in_range = False

        for v in self._versions:
            if v.version == from_version:
                in_range = True
                continue

            if in_range:
                changes.extend(v.changes)

            if v.version == to_version:
                break

        return changes

    # ========== 文件操作 ==========

    def save(self, path: Path | str, format: str = "yaml") -> None:
        """保存文档.

        Args:
            path: 文件路径
            format: 格式 (yaml 或 json)
        """
        if not self._doc:
            raise ValueError("未加载文档")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._doc.model_dump()

        with open(path, "w", encoding="utf-8") as f:
            if format == "yaml":
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"保存文档: {path}")

    def _save_version(self, version: DocVersion) -> None:
        """保存版本到文件."""
        if not self._history_dir:
            return

        self._history_dir.mkdir(parents=True, exist_ok=True)
        version_file = self._history_dir / f"v{version.version.replace('.', '_')}.json"

        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(version.to_dict(), f, ensure_ascii=False, indent=2)

    def _remove_version_file(self, version: DocVersion) -> None:
        """删除版本文件."""
        if not self._history_dir:
            return

        version_file = self._history_dir / f"v{version.version.replace('.', '_')}.json"
        if version_file.exists():
            version_file.unlink()

    def _load_version_history(self) -> None:
        """加载版本历史."""
        if not self._history_dir or not self._history_dir.exists():
            return

        for version_file in sorted(self._history_dir.glob("v*.json")):
            try:
                with open(version_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                version = DocVersion(
                    version=data["version"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    doc_hash=data["doc_hash"],
                    changes=[
                        ChangeRecord(
                            change_type=ChangeType(c["change_type"]),
                            timestamp=datetime.fromisoformat(c["timestamp"]),
                            description=c["description"],
                            path=c["path"],
                            old_value=c.get("old_value"),
                            new_value=c.get("new_value"),
                            reason=c.get("reason", ""),
                        )
                        for c in data.get("changes", [])
                    ],
                )
                self._versions.append(version)
            except Exception as e:
                logger.warning(f"加载版本文件失败: {version_file}, {e}")

        if self._versions:
            self._current_version = self._versions[-1].version

    # ========== 便捷方法 ==========

    def generate_next_task_id(self) -> str:
        """生成下一个任务 ID.

        Returns:
            新任务 ID
        """
        if not self._doc:
            return "task-001"

        max_num = 0
        for module in self._doc.modules:
            for task in module.sub_tasks:
                if task.id.startswith("task-"):
                    try:
                        num = int(task.id[5:])
                        max_num = max(max_num, num)
                    except ValueError:
                        pass

        return f"task-{max_num + 1:03d}"

    def generate_next_module_id(self, prefix: str = "mod-new") -> str:
        """生成下一个模块 ID.

        Args:
            prefix: ID 前缀

        Returns:
            新模块 ID
        """
        if not self._doc:
            return f"{prefix}-1"

        existing = {m.id for m in self._doc.modules}
        counter = 1
        while f"{prefix}-{counter}" in existing:
            counter += 1

        return f"{prefix}-{counter}"

    def get_summary(self) -> dict[str, Any]:
        """获取文档摘要.

        Returns:
            摘要信息
        """
        if not self._doc:
            return {"error": "未加载文档"}

        total_tasks = sum(len(m.sub_tasks) for m in self._doc.modules)
        total_minutes = sum(
            t.estimated_minutes
            for m in self._doc.modules
            for t in m.sub_tasks
        )

        return {
            "project_name": self._doc.project_name,
            "version": self._current_version,
            "module_count": len(self._doc.modules),
            "task_count": total_tasks,
            "estimated_hours": round(total_minutes / 60, 1),
            "pending_changes": len(self._pending_changes),
            "total_versions": len(self._versions),
        }
