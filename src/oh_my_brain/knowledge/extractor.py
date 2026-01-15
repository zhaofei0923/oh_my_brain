"""自动知识提取器.

从代码、文档和任务结果中自动提取知识。
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from oh_my_brain.knowledge import (
    BugFixEntry,
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeType,
)

logger = logging.getLogger(__name__)


class ExtractionSource(str, Enum):
    """提取来源."""

    CODE = "code"           # 代码文件
    DOCSTRING = "docstring" # 文档字符串
    COMMENT = "comment"     # 代码注释
    COMMIT = "commit"       # Git 提交
    TASK_RESULT = "task"    # 任务结果
    MANUAL = "manual"       # 手动添加


@dataclass
class ExtractedKnowledge:
    """提取的知识."""

    type: KnowledgeType
    title: str
    content: str
    source: ExtractionSource
    source_file: str = ""
    line_number: int = 0
    confidence: float = 1.0
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class CodeKnowledgeExtractor:
    """代码知识提取器.

    从代码文件中提取：
    1. TODO/FIXME/HACK 注释
    2. 文档字符串中的说明
    3. 异常处理模式
    4. 设计模式使用
    """

    # 匹配模式
    TODO_PATTERN = re.compile(
        r'#\s*(TODO|FIXME|HACK|NOTE|XXX|BUG)[\s:]+(.+?)(?:\n|$)',
        re.IGNORECASE
    )

    DOCSTRING_PATTERN = re.compile(
        r'"""(.+?)"""',
        re.DOTALL
    )

    EXCEPTION_PATTERN = re.compile(
        r'except\s+(\w+(?:\s*,\s*\w+)*)\s*(?:as\s+\w+)?:\s*\n\s*(.+?)(?=\n\s*(?:except|else|finally|\S)|\Z)',
        re.DOTALL
    )

    def __init__(self, min_confidence: float = 0.5):
        """初始化.

        Args:
            min_confidence: 最小置信度阈值
        """
        self._min_confidence = min_confidence

    def extract_from_file(self, file_path: Path) -> list[ExtractedKnowledge]:
        """从文件提取知识.

        Args:
            file_path: 文件路径

        Returns:
            提取的知识列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return []

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        results = []

        # 根据文件类型选择提取方法
        if file_path.suffix in [".py"]:
            results.extend(self._extract_from_python(content, str(file_path)))
        elif file_path.suffix in [".js", ".ts", ".jsx", ".tsx"]:
            results.extend(self._extract_from_javascript(content, str(file_path)))
        elif file_path.suffix in [".md", ".rst"]:
            results.extend(self._extract_from_markdown(content, str(file_path)))

        return results

    def _extract_from_python(
        self,
        content: str,
        source_file: str,
    ) -> list[ExtractedKnowledge]:
        """从 Python 代码提取知识."""
        results = []

        # 提取 TODO/FIXME 等注释
        for match in self.TODO_PATTERN.finditer(content):
            tag = match.group(1).upper()
            text = match.group(2).strip()

            knowledge_type = {
                "TODO": KnowledgeType.TIP,
                "FIXME": KnowledgeType.BUG_FIX,
                "HACK": KnowledgeType.ANTI_PATTERN,
                "NOTE": KnowledgeType.TIP,
                "XXX": KnowledgeType.LESSON,
                "BUG": KnowledgeType.BUG_FIX,
            }.get(tag, KnowledgeType.TIP)

            line_num = content[:match.start()].count("\n") + 1

            results.append(ExtractedKnowledge(
                type=knowledge_type,
                title=f"{tag}: {text[:50]}",
                content=text,
                source=ExtractionSource.COMMENT,
                source_file=source_file,
                line_number=line_num,
                confidence=0.6,
                tags=[tag.lower(), "auto-extracted"],
            ))

        # 提取异常处理模式
        for match in self.EXCEPTION_PATTERN.finditer(content):
            exception_types = match.group(1)
            handling_code = match.group(2).strip()

            if len(handling_code) > 20:  # 忽略简单的 pass
                line_num = content[:match.start()].count("\n") + 1

                results.append(ExtractedKnowledge(
                    type=KnowledgeType.PATTERN,
                    title=f"异常处理: {exception_types}",
                    content=f"异常类型: {exception_types}\n处理方式:\n{handling_code}",
                    source=ExtractionSource.CODE,
                    source_file=source_file,
                    line_number=line_num,
                    confidence=0.5,
                    tags=["exception", "error-handling", "auto-extracted"],
                ))

        # 提取文档字符串中的重要说明
        results.extend(self._extract_docstring_knowledge(content, source_file))

        return results

    def _extract_from_javascript(
        self,
        content: str,
        source_file: str,
    ) -> list[ExtractedKnowledge]:
        """从 JavaScript/TypeScript 代码提取知识."""
        results = []

        # 匹配 // TODO 和 /* TODO */
        js_todo_pattern = re.compile(
            r'(?://|/\*)\s*(TODO|FIXME|HACK|NOTE|XXX|BUG)[\s:]+(.+?)(?:\n|\*/|$)',
            re.IGNORECASE
        )

        for match in js_todo_pattern.finditer(content):
            tag = match.group(1).upper()
            text = match.group(2).strip()

            knowledge_type = {
                "TODO": KnowledgeType.TIP,
                "FIXME": KnowledgeType.BUG_FIX,
                "HACK": KnowledgeType.ANTI_PATTERN,
            }.get(tag, KnowledgeType.TIP)

            line_num = content[:match.start()].count("\n") + 1

            results.append(ExtractedKnowledge(
                type=knowledge_type,
                title=f"{tag}: {text[:50]}",
                content=text,
                source=ExtractionSource.COMMENT,
                source_file=source_file,
                line_number=line_num,
                confidence=0.6,
                tags=[tag.lower(), "javascript", "auto-extracted"],
            ))

        return results

    def _extract_from_markdown(
        self,
        content: str,
        source_file: str,
    ) -> list[ExtractedKnowledge]:
        """从 Markdown 文档提取知识."""
        results = []

        # 提取标题和内容
        section_pattern = re.compile(r'^(#{1,3})\s+(.+?)$', re.MULTILINE)
        sections = []

        matches = list(section_pattern.finditer(content))
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            # 获取内容直到下一个标题
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            if len(section_content) > 50:  # 忽略过短内容
                sections.append((level, title, section_content))

        # 识别最佳实践、注意事项等
        keywords = {
            "best practice": KnowledgeType.BEST_PRACTICE,
            "最佳实践": KnowledgeType.BEST_PRACTICE,
            "注意": KnowledgeType.LESSON,
            "warning": KnowledgeType.LESSON,
            "常见问题": KnowledgeType.FAQ,
            "faq": KnowledgeType.FAQ,
            "tip": KnowledgeType.TIP,
            "技巧": KnowledgeType.TIP,
        }

        for level, title, section_content in sections:
            title_lower = title.lower()
            knowledge_type = None

            for keyword, kt in keywords.items():
                if keyword in title_lower:
                    knowledge_type = kt
                    break

            if knowledge_type:
                results.append(ExtractedKnowledge(
                    type=knowledge_type,
                    title=title,
                    content=section_content[:500],
                    source=ExtractionSource.DOCSTRING,
                    source_file=source_file,
                    confidence=0.7,
                    tags=["documentation", "auto-extracted"],
                ))

        return results

    def _extract_docstring_knowledge(
        self,
        content: str,
        source_file: str,
    ) -> list[ExtractedKnowledge]:
        """从文档字符串提取知识."""
        results = []

        # 查找包含特殊标记的文档字符串
        keywords = {
            "Note:": KnowledgeType.TIP,
            "Warning:": KnowledgeType.LESSON,
            "Example:": KnowledgeType.PATTERN,
            "Raises:": KnowledgeType.PATTERN,
            "注意:": KnowledgeType.LESSON,
            "示例:": KnowledgeType.PATTERN,
        }

        for match in self.DOCSTRING_PATTERN.finditer(content):
            docstring = match.group(1)
            line_num = content[:match.start()].count("\n") + 1

            for keyword, knowledge_type in keywords.items():
                if keyword in docstring:
                    # 提取关键字后的内容
                    idx = docstring.find(keyword)
                    # 找到下一个关键字或结尾
                    end_idx = len(docstring)
                    for other_kw in keywords:
                        other_idx = docstring.find(other_kw, idx + len(keyword))
                        if other_idx > 0:
                            end_idx = min(end_idx, other_idx)

                    snippet = docstring[idx:end_idx].strip()

                    if len(snippet) > 20:
                        results.append(ExtractedKnowledge(
                            type=knowledge_type,
                            title=f"Docstring: {keyword.rstrip(':')}",
                            content=snippet,
                            source=ExtractionSource.DOCSTRING,
                            source_file=source_file,
                            line_number=line_num,
                            confidence=0.5,
                            tags=["docstring", "auto-extracted"],
                        ))

        return results


class GitCommitExtractor:
    """Git 提交知识提取器.

    从 Git 提交历史中提取：
    1. Bug 修复提交
    2. 重构提交
    3. 功能实现模式
    """

    BUG_FIX_PATTERNS = [
        r'fix(?:ed|es|ing)?\s*[:#]?\s*(.+)',
        r'bug(?:fix)?[:#]?\s*(.+)',
        r'resolve[sd]?\s*[:#]?\s*(.+)',
        r'修复\s*[:#]?\s*(.+)',
    ]

    REFACTOR_PATTERNS = [
        r'refactor(?:ed|ing)?\s*[:#]?\s*(.+)',
        r'重构\s*[:#]?\s*(.+)',
        r'cleanup\s*[:#]?\s*(.+)',
    ]

    def __init__(self, repo_path: Path | None = None):
        """初始化.

        Args:
            repo_path: Git 仓库路径
        """
        self._repo_path = repo_path or Path.cwd()

    async def extract_from_history(
        self,
        max_commits: int = 100,
    ) -> list[ExtractedKnowledge]:
        """从 Git 历史提取知识.

        Args:
            max_commits: 最大提交数

        Returns:
            提取的知识列表
        """
        import subprocess

        results = []

        try:
            # 获取提交历史
            output = subprocess.check_output(
                ["git", "log", f"-{max_commits}", "--pretty=format:%H|%s|%b|||"],
                cwd=self._repo_path,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("无法读取 Git 历史")
            return results

        commits = output.split("|||")

        for commit_data in commits:
            if not commit_data.strip():
                continue

            parts = commit_data.strip().split("|", 2)
            if len(parts) < 2:
                continue

            commit_hash = parts[0]
            subject = parts[1]
            body = parts[2] if len(parts) > 2 else ""

            # 检查是否是 Bug 修复
            for pattern in self.BUG_FIX_PATTERNS:
                match = re.search(pattern, subject, re.IGNORECASE)
                if match:
                    results.append(ExtractedKnowledge(
                        type=KnowledgeType.BUG_FIX,
                        title=f"Bug 修复: {match.group(1)[:50]}",
                        content=f"提交: {commit_hash[:8]}\n{subject}\n\n{body}",
                        source=ExtractionSource.COMMIT,
                        source_file=commit_hash,
                        confidence=0.7,
                        tags=["git", "bugfix", "auto-extracted"],
                    ))
                    break

            # 检查是否是重构
            for pattern in self.REFACTOR_PATTERNS:
                match = re.search(pattern, subject, re.IGNORECASE)
                if match:
                    results.append(ExtractedKnowledge(
                        type=KnowledgeType.PATTERN,
                        title=f"重构: {match.group(1)[:50]}",
                        content=f"提交: {commit_hash[:8]}\n{subject}\n\n{body}",
                        source=ExtractionSource.COMMIT,
                        source_file=commit_hash,
                        confidence=0.6,
                        tags=["git", "refactor", "auto-extracted"],
                    ))
                    break

        return results


class KnowledgeExtractor:
    """统一知识提取器.

    整合各种提取器，提供统一接口。
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        min_confidence: float = 0.5,
    ):
        """初始化.

        Args:
            knowledge_base: 知识库
            min_confidence: 最小置信度
        """
        self._kb = knowledge_base
        self._min_confidence = min_confidence

        self._code_extractor = CodeKnowledgeExtractor(min_confidence)
        self._git_extractor = GitCommitExtractor()

    async def extract_from_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> list[ExtractedKnowledge]:
        """从目录提取知识.

        Args:
            directory: 目录路径
            recursive: 是否递归
            extensions: 文件扩展名

        Returns:
            提取的知识列表
        """
        directory = Path(directory)
        if not directory.is_dir():
            return []

        extensions = extensions or [".py", ".js", ".ts", ".md"]
        results = []

        # 获取文件列表
        files = []
        for ext in extensions:
            if recursive:
                files.extend(directory.rglob(f"*{ext}"))
            else:
                files.extend(directory.glob(f"*{ext}"))

        # 提取知识
        for file_path in files:
            # 跳过一些目录
            if any(p in str(file_path) for p in ["node_modules", ".git", "__pycache__", ".venv"]):
                continue

            extracted = self._code_extractor.extract_from_file(file_path)
            results.extend(extracted)

        # 过滤低置信度
        results = [r for r in results if r.confidence >= self._min_confidence]

        logger.info(f"从 {len(files)} 个文件中提取了 {len(results)} 条知识")
        return results

    async def extract_from_git(
        self,
        repo_path: Path | None = None,
        max_commits: int = 100,
    ) -> list[ExtractedKnowledge]:
        """从 Git 历史提取知识.

        Args:
            repo_path: 仓库路径
            max_commits: 最大提交数

        Returns:
            提取的知识列表
        """
        if repo_path:
            self._git_extractor = GitCommitExtractor(repo_path)

        results = await self._git_extractor.extract_from_history(max_commits)

        # 过滤低置信度
        results = [r for r in results if r.confidence >= self._min_confidence]

        logger.info(f"从 Git 历史中提取了 {len(results)} 条知识")
        return results

    def save_to_knowledge_base(
        self,
        extracted: list[ExtractedKnowledge],
        deduplicate: bool = True,
    ) -> int:
        """保存提取的知识到知识库.

        Args:
            extracted: 提取的知识列表
            deduplicate: 是否去重

        Returns:
            保存的条目数
        """
        if not self._kb:
            logger.warning("未配置知识库")
            return 0

        saved = 0
        seen_titles = set()

        for item in extracted:
            # 去重
            if deduplicate:
                if item.title in seen_titles:
                    continue
                seen_titles.add(item.title)

            # 创建知识条目
            import uuid
            entry_id = f"kb-auto-{uuid.uuid4().hex[:8]}"

            if item.type == KnowledgeType.BUG_FIX:
                entry = BugFixEntry(
                    id=entry_id,
                    title=item.title,
                    description=item.content,
                    tags=item.tags,
                    source=f"{item.source.value}:{item.source_file}",
                    confidence=item.confidence,
                )
            else:
                entry = KnowledgeEntry(
                    id=entry_id,
                    type=item.type,
                    title=item.title,
                    description=item.content,
                    tags=item.tags,
                    source=f"{item.source.value}:{item.source_file}",
                    confidence=item.confidence,
                )

            self._kb.add(entry)
            saved += 1

        if saved > 0:
            self._kb.save()
            logger.info(f"保存 {saved} 条知识到知识库")

        return saved


async def auto_extract_knowledge(
    source_dir: Path,
    knowledge_base: KnowledgeBase,
    include_git: bool = True,
    min_confidence: float = 0.5,
) -> int:
    """自动提取知识的便捷函数.

    Args:
        source_dir: 源代码目录
        knowledge_base: 知识库
        include_git: 是否包含 Git 历史
        min_confidence: 最小置信度

    Returns:
        提取的知识数量
    """
    extractor = KnowledgeExtractor(
        knowledge_base=knowledge_base,
        min_confidence=min_confidence,
    )

    all_extracted = []

    # 从代码提取
    code_extracted = await extractor.extract_from_directory(source_dir)
    all_extracted.extend(code_extracted)

    # 从 Git 提取
    if include_git:
        git_extracted = await extractor.extract_from_git(source_dir)
        all_extracted.extend(git_extracted)

    # 保存到知识库
    saved = extractor.save_to_knowledge_base(all_extracted)

    return saved
