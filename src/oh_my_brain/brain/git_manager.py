"""Git管理器.

管理Git分支、提交、合并操作。
"""

import logging
from pathlib import Path
from typing import Any

from git import GitCommandError, Repo

logger = logging.getLogger(__name__)


class GitManager:
    """Git管理器.

    功能：
    1. 为每个任务创建独立分支
    2. 管理分支合并
    3. 处理合并冲突
    """

    def __init__(
        self,
        repo_path: Path | str | None = None,
        base_branch: str = "develop",
        branch_prefix: str = "agent/",
    ):
        self._repo_path = Path(repo_path) if repo_path else None
        self._repo: Repo | None = None
        self._base_branch = base_branch
        self._branch_prefix = branch_prefix

    def init_repo(self, path: Path | str) -> None:
        """初始化仓库连接.

        Args:
            path: 仓库路径
        """
        self._repo_path = Path(path)
        try:
            self._repo = Repo(self._repo_path)
            logger.info(f"Git repo initialized: {self._repo_path}")
        except Exception as e:
            logger.error(f"Failed to init git repo: {e}")
            self._repo = None

    def is_initialized(self) -> bool:
        """检查是否已初始化."""
        return self._repo is not None

    def create_task_branch(self, task_id: str) -> str | None:
        """为任务创建分支.

        Args:
            task_id: 任务ID

        Returns:
            分支名称，失败返回None
        """
        if not self._repo:
            return None

        branch_name = f"{self._branch_prefix}{task_id}"

        try:
            # 确保在最新的base分支上
            self._repo.git.checkout(self._base_branch)
            self._repo.git.pull("origin", self._base_branch)

            # 创建并切换到新分支
            self._repo.git.checkout("-b", branch_name)

            logger.info(f"Created branch: {branch_name}")
            return branch_name

        except GitCommandError as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return None

    def switch_branch(self, branch_name: str) -> bool:
        """切换分支.

        Args:
            branch_name: 分支名称

        Returns:
            是否成功
        """
        if not self._repo:
            return False

        try:
            self._repo.git.checkout(branch_name)
            return True
        except GitCommandError as e:
            logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False

    def commit(self, message: str, files: list[str] | None = None) -> str | None:
        """提交更改.

        Args:
            message: 提交信息
            files: 要提交的文件列表，None则提交所有更改

        Returns:
            commit SHA，失败返回None
        """
        if not self._repo:
            return None

        try:
            if files:
                self._repo.index.add(files)
            else:
                self._repo.git.add("-A")

            # 检查是否有更改
            if not self._repo.is_dirty() and not self._repo.untracked_files:
                logger.info("No changes to commit")
                return None

            commit = self._repo.index.commit(message)
            logger.info(f"Committed: {commit.hexsha[:8]} - {message}")
            return commit.hexsha

        except GitCommandError as e:
            logger.error(f"Failed to commit: {e}")
            return None

    def push_branch(self, branch_name: str) -> bool:
        """推送分支到远程.

        Args:
            branch_name: 分支名称

        Returns:
            是否成功
        """
        if not self._repo:
            return False

        try:
            self._repo.git.push("-u", "origin", branch_name)
            logger.info(f"Pushed branch: {branch_name}")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to push branch {branch_name}: {e}")
            return False

    def merge_branch(
        self,
        source_branch: str,
        target_branch: str | None = None,
        no_ff: bool = True,
    ) -> tuple[bool, str]:
        """合并分支.

        Args:
            source_branch: 源分支
            target_branch: 目标分支，默认为base_branch
            no_ff: 是否强制创建合并提交

        Returns:
            (是否成功, 信息)
        """
        if not self._repo:
            return False, "Repository not initialized"

        target_branch = target_branch or self._base_branch

        try:
            # 切换到目标分支
            self._repo.git.checkout(target_branch)
            self._repo.git.pull("origin", target_branch)

            # 执行合并
            merge_args = ["--no-ff"] if no_ff else []
            self._repo.git.merge(source_branch, *merge_args)

            logger.info(f"Merged {source_branch} into {target_branch}")
            return True, "Merge successful"

        except GitCommandError as e:
            error_msg = str(e)
            if "CONFLICT" in error_msg:
                logger.warning(f"Merge conflict: {source_branch} -> {target_branch}")
                # 中止合并，等待人工处理
                self._repo.git.merge("--abort")
                return False, "Merge conflict detected, manual resolution required"
            else:
                logger.error(f"Merge failed: {e}")
                return False, f"Merge failed: {error_msg}"

    def delete_branch(self, branch_name: str, force: bool = False) -> bool:
        """删除分支.

        Args:
            branch_name: 分支名称
            force: 是否强制删除

        Returns:
            是否成功
        """
        if not self._repo:
            return False

        try:
            # 确保不在要删除的分支上
            if self._repo.active_branch.name == branch_name:
                self._repo.git.checkout(self._base_branch)

            delete_flag = "-D" if force else "-d"
            self._repo.git.branch(delete_flag, branch_name)
            logger.info(f"Deleted branch: {branch_name}")
            return True

        except GitCommandError as e:
            logger.error(f"Failed to delete branch {branch_name}: {e}")
            return False

    def get_branch_diff(
        self,
        branch_name: str,
        base_branch: str | None = None,
    ) -> str:
        """获取分支差异.

        Args:
            branch_name: 分支名称
            base_branch: 基准分支

        Returns:
            diff内容
        """
        if not self._repo:
            return ""

        base_branch = base_branch or self._base_branch

        try:
            return self._repo.git.diff(f"{base_branch}...{branch_name}")
        except GitCommandError as e:
            logger.error(f"Failed to get diff: {e}")
            return ""

    def get_branch_commits(
        self,
        branch_name: str,
        base_branch: str | None = None,
        max_count: int = 50,
    ) -> list[dict[str, Any]]:
        """获取分支提交历史.

        Args:
            branch_name: 分支名称
            base_branch: 基准分支
            max_count: 最大提交数

        Returns:
            提交列表
        """
        if not self._repo:
            return []

        base_branch = base_branch or self._base_branch

        try:
            commits = list(
                self._repo.iter_commits(
                    f"{base_branch}..{branch_name}",
                    max_count=max_count,
                )
            )

            return [
                {
                    "sha": c.hexsha,
                    "message": c.message.strip(),
                    "author": str(c.author),
                    "date": c.committed_datetime.isoformat(),
                }
                for c in commits
            ]

        except GitCommandError as e:
            logger.error(f"Failed to get commits: {e}")
            return []

    def get_current_branch(self) -> str | None:
        """获取当前分支名称."""
        if not self._repo:
            return None
        try:
            return self._repo.active_branch.name
        except Exception:
            return None

    def list_task_branches(self) -> list[str]:
        """列出所有任务分支."""
        if not self._repo:
            return []

        try:
            branches = [
                ref.name
                for ref in self._repo.references
                if ref.name.startswith(self._branch_prefix)
            ]
            return branches
        except Exception as e:
            logger.error(f"Failed to list branches: {e}")
            return []

    def get_status(self) -> dict[str, Any]:
        """获取Git状态."""
        if not self._repo:
            return {"initialized": False}

        return {
            "initialized": True,
            "repo_path": str(self._repo_path),
            "current_branch": self.get_current_branch(),
            "is_dirty": self._repo.is_dirty(),
            "task_branches": self.list_task_branches(),
        }
