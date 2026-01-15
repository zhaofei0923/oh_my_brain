"""Brain组件测试."""

import pytest

from oh_my_brain.brain.doc_parser import DocParser
from oh_my_brain.brain.safety_checker import SafetyChecker


class TestDocParser:
    """测试文档解析器."""

    def test_parse_dict(self):
        """测试字典解析."""
        data = {
            "project": {
                "name": "Test",
                "description": "Test project",
                "tech_stack": {
                    "language": "Python",
                },
            },
            "modules": [
                {
                    "id": "mod-1",
                    "name": "Module 1",
                    "description": "Test module",
                    "acceptance_criteria": "Test passed",
                    "sub_tasks": [
                        {
                            "id": "task-001",
                            "name": "Task 1",
                            "type": "feature",
                            "description": "Do something",
                            "requirements": "Requirements",
                        }
                    ],
                }
            ],
        }

        doc = DocParser.parse_dict(data)
        assert doc.project.name == "Test"

    def test_parse_invalid_dict(self):
        """测试无效字典."""
        data = {"invalid": "data"}

        with pytest.raises(ValueError):
            DocParser.parse_dict(data)

    def test_generate_template(self):
        """测试模板生成."""
        template = DocParser.generate_template()

        assert "project:" in template
        assert "modules:" in template
        assert "sub_tasks:" in template

    def test_export_json_schema(self):
        """测试JSON Schema导出."""
        schema = DocParser.export_json_schema()

        assert "properties" in schema
        assert "project" in schema["properties"]
        assert "modules" in schema["properties"]


class TestSafetyChecker:
    """测试安全检查器."""

    def test_dangerous_bash_command(self):
        """测试危险bash命令."""
        checker = SafetyChecker()

        result = checker.check("bash", "rm -rf /")
        assert not result.approved
        assert "Dangerous" in result.reason

    def test_safe_bash_command(self):
        """测试安全bash命令."""
        checker = SafetyChecker()

        result = checker.check("bash", "ls -la")
        assert result.approved

    def test_sudo_command_rejected(self):
        """测试sudo命令被拒绝."""
        checker = SafetyChecker()

        result = checker.check("bash", "sudo apt update")
        assert not result.approved
        assert "sudo" in result.reason.lower()

    def test_protected_path_write(self):
        """测试受保护路径写入."""
        checker = SafetyChecker()

        result = checker.check("write", "/etc/passwd")
        assert not result.approved

    def test_tmp_path_allowed(self):
        """测试/tmp路径允许."""
        checker = SafetyChecker()

        result = checker.check("write", "/tmp/test.txt")
        assert result.approved

    def test_disabled_checker(self):
        """测试禁用的检查器."""
        checker = SafetyChecker(enabled=False)

        result = checker.check("bash", "rm -rf /")
        assert result.approved

    def test_custom_dangerous_pattern(self):
        """测试自定义危险模式."""
        checker = SafetyChecker()
        checker.add_dangerous_pattern(r"format\s+c:")

        result = checker.check("bash", "format c:")
        assert not result.approved

    def test_custom_protected_path(self):
        """测试自定义受保护路径."""
        checker = SafetyChecker()
        checker.add_protected_path("/my/protected")

        result = checker.check("write", "/my/protected/file.txt")
        assert not result.approved
