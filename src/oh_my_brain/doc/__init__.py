"""开发文档管理模块.

提供开发文档的生成、验证和更新功能。
"""

from oh_my_brain.doc.generator import (
    DocGenerator,
    GenerationMode,
    ProjectType,
    load_dev_doc,
    save_dev_doc,
)
from oh_my_brain.doc.updater import (
    ChangeRecord,
    ChangeType,
    DocUpdater,
    DocVersion,
)
from oh_my_brain.doc.validator import (
    DocValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    validate_dev_doc_file,
    validate_dev_doc_yaml,
)

__all__ = [
    # Generator
    "DocGenerator",
    "GenerationMode",
    "ProjectType",
    "save_dev_doc",
    "load_dev_doc",
    # Validator
    "DocValidator",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "validate_dev_doc_file",
    "validate_dev_doc_yaml",
    # Updater
    "DocUpdater",
    "DocVersion",
    "ChangeRecord",
    "ChangeType",
]
