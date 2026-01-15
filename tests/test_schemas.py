"""Schema测试."""

from oh_my_brain.schemas.dev_doc import DevDoc
from oh_my_brain.schemas.task import Task, TaskStatus, TaskType


class TestDevDoc:
    """测试DevDoc模型."""

    def test_minimal_dev_doc(self):
        """测试最小化DevDoc."""
        data = {
            "project": {
                "name": "Test Project",
                "description": "A test project",
            },
            "modules": [
                {
                    "id": "mod-1",
                    "name": "Module 1",
                    "sub_tasks": [
                        {
                            "id": "task-1",
                            "name": "Task 1",
                            "type": "feature",
                            "description": "A task",
                            "requirements": "Do something",
                        }
                    ],
                }
            ],
        }

        doc = DevDoc.model_validate(data)
        assert doc.project.name == "Test Project"
        assert len(doc.modules) == 1
        assert len(doc.modules[0].sub_tasks) == 1

    def test_full_dev_doc(self):
        """测试完整DevDoc."""
        data = {
            "project": {
                "name": "Full Project",
                "version": "1.0.0",
                "description": "A full project",
                "tech_stack": {
                    "language": "Python",
                    "framework": "FastAPI",
                    "database": "PostgreSQL",
                    "other": ["Redis", "Docker"],
                },
            },
            "modules": [
                {
                    "id": "mod-auth",
                    "name": "Authentication",
                    "description": "User authentication module",
                    "priority": 1,
                    "dependencies": [],
                    "acceptance_criteria": "Users can login",
                    "sub_tasks": [
                        {
                            "id": "task-001",
                            "name": "Create login endpoint",
                            "type": "feature",
                            "description": "Implement login",
                            "estimated_minutes": 30,
                            "files_involved": ["src/auth/login.py"],
                            "requirements": "POST /login endpoint",
                        }
                    ],
                }
            ],
        }

        doc = DevDoc.model_validate(data)
        assert doc.project.tech_stack.language == "Python"
        assert doc.modules[0].priority == 1


class TestTask:
    """测试Task模型."""

    def test_task_creation(self):
        """测试Task创建."""
        task = Task(
            id="task-1",
            name="Test Task",
            type=TaskType.FEATURE,
            description="A test task",
            requirements="Do something",
        )

        assert task.id == "task-1"
        assert task.status == TaskStatus.PENDING
        assert task.type == TaskType.FEATURE

    def test_task_status_transitions(self):
        """测试Task状态转换."""
        task = Task(
            id="task-1",
            name="Test Task",
            type=TaskType.FEATURE,
            description="A test task",
            requirements="Do something",
        )

        assert task.status == TaskStatus.PENDING
        task.status = TaskStatus.RUNNING
        assert task.status == TaskStatus.RUNNING
        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED
