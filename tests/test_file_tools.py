"""
Unit tests for Sandboxed File Management Tools (tools/file_tools.py).
Validates path jail integrity, traversal protection, file CRUD operations,
and destructive action approval gates.
"""

from pathlib import Path
import pytest

from config import Config, load_config
from tools.file_tools import (
    create_file,
    delete_file,
    list_files,
    read_file,
    resolve_safe_path
)


@pytest.fixture
def test_config(tmp_path):
    """Provides an isolated configuration pointing to a temporary workspace."""
    workspace = tmp_path / "sandbox_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return Config(
        groq_api_key="mock_key",
        model_name="llama-3.3-70b-versatile",
        max_iterations=10,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=workspace,
        project_root=tmp_path
    )


class TestFileCreation:
    """Tests for create_file functionality and overwrite safety."""

    def test_create_file_success(self, test_config):
        result = create_file("hello.py", "print('Hello, World!')", config=test_config)
        assert result["success"] is True
        assert result["filename"] == "hello.py"
        assert result["bytes_written"] > 0

        target_file = test_config.workspace_dir / "hello.py"
        assert target_file.exists()
        assert target_file.read_text(encoding="utf-8") == "print('Hello, World!')"

    def test_create_file_nested_subdirectories(self, test_config):
        result = create_file("src/modules/math_utils.py", "def add(a, b): return a + b", config=test_config)
        assert result["success"] is True

        target_file = test_config.workspace_dir / "src" / "modules" / "math_utils.py"
        assert target_file.exists()
        assert "def add" in target_file.read_text(encoding="utf-8")

    def test_create_file_prevent_accidental_overwrite(self, test_config):
        # Initial creation
        create_file("data.txt", "Original Data", config=test_config)

        # Attempt to overwrite without setting overwrite=True
        result = create_file("data.txt", "New Data", overwrite=False, config=test_config)
        assert result["success"] is False
        assert "already exists" in result["error"]

        # Content must remain unchanged
        assert (test_config.workspace_dir / "data.txt").read_text() == "Original Data"

    def test_create_file_overwrite_without_approval_blocked(self, test_config):
        create_file("data.txt", "Original Data", config=test_config)

        # Overwrite=True but approved=False
        result = create_file("data.txt", "New Data", overwrite=True, approved=False, config=test_config)
        assert result["success"] is False
        assert result.get("approval_required") is True
        assert "requires explicit user approval" in result["error"]

        # Content remains unchanged
        assert (test_config.workspace_dir / "data.txt").read_text() == "Original Data"

    def test_create_file_overwrite_with_approval_succeeds(self, test_config):
        create_file("data.txt", "Original Data", config=test_config)

        # Overwrite=True AND approved=True
        result = create_file("data.txt", "Replaced Data", overwrite=True, approved=True, config=test_config)
        assert result["success"] is True
        assert (test_config.workspace_dir / "data.txt").read_text() == "Replaced Data"


class TestFileReading:
    """Tests for read_file operations and edge cases."""

    def test_read_file_success(self, test_config):
        content = "line 1\nline 2\nline 3"
        create_file("multiline.txt", content, config=test_config)

        result = read_file("multiline.txt", config=test_config)
        assert result["success"] is True
        assert result["content"] == content
        assert result["line_count"] == 3
        assert result["size_bytes"] == len(content.encode("utf-8"))

    def test_read_nonexistent_file(self, test_config):
        result = read_file("ghost.txt", config=test_config)
        assert result["success"] is False
        assert "File not found" in result["error"]

    def test_read_directory_fails_gracefully(self, test_config):
        sub = test_config.workspace_dir / "subfolder"
        sub.mkdir()

        result = read_file("subfolder", config=test_config)
        assert result["success"] is False
        assert "is a directory" in result["error"]


class TestFileListing:
    """Tests for list_files directory inspection."""

    def test_list_files_empty_workspace(self, test_config):
        result = list_files(config=test_config)
        assert result["success"] is True
        assert result["total_items"] == 0
        assert result["entries"] == []

    def test_list_files_populated(self, test_config):
        create_file("script1.py", "print(1)", config=test_config)
        create_file("sub/script2.py", "print(2)", config=test_config)

        result = list_files(config=test_config)
        assert result["success"] is True
        names = [entry["name"] for entry in result["entries"]]
        assert "script1.py" in names
        assert "sub" in names

        # Inspect subfolder listing
        sub_result = list_files("sub", config=test_config)
        assert sub_result["success"] is True
        assert len(sub_result["entries"]) == 1
        assert sub_result["entries"][0]["name"] == "script2.py"


class TestFileDeletion:
    """Tests for delete_file approval enforcement."""

    def test_delete_file_unapproved_blocked(self, test_config):
        create_file("target.txt", "To be deleted", config=test_config)

        result = delete_file("target.txt", approved=False, config=test_config)
        assert result["success"] is False
        assert result.get("approval_required") is True
        assert "requires explicit user approval" in result["error"]
        assert (test_config.workspace_dir / "target.txt").exists()

    def test_delete_file_approved_succeeds(self, test_config):
        create_file("target.txt", "To be deleted", config=test_config)

        result = delete_file("target.txt", approved=True, config=test_config)
        assert result["success"] is True
        assert not (test_config.workspace_dir / "target.txt").exists()

    def test_delete_nonexistent_file(self, test_config):
        result = delete_file("missing.txt", approved=True, config=test_config)
        assert result["success"] is False
        assert "nonexistent file" in result["error"]


class TestPathTraversalSecurity:
    """Tests guaranteeing the sandbox path jail cannot be escaped."""

    def test_relative_path_traversal_blocked(self, test_config):
        # Attempt relative traversal: ../../escape.txt
        result = create_file("../../escape.txt", "malicious", config=test_config)
        assert result["success"] is False
        assert "Path traversal detected" in result["error"]

    def test_read_path_traversal_blocked(self, test_config):
        result = read_file("../outside_file.txt", config=test_config)
        assert result["success"] is False
        assert "Path traversal detected" in result["error"]

    def test_delete_path_traversal_blocked(self, test_config):
        result = delete_file("../../sensitive.txt", approved=True, config=test_config)
        assert result["success"] is False
        assert "Path traversal detected" in result["error"]

    def test_absolute_path_outside_workspace_blocked(self, test_config, tmp_path):
        outside_file = tmp_path / "outside_dir" / "secret.txt"
        outside_file.parent.mkdir(parents=True, exist_ok=True)
        outside_file.write_text("secret")

        result = read_file(str(outside_file), config=test_config)
        assert result["success"] is False
        assert ("Path traversal detected" in result["error"] or "Absolute path" in result["error"])

    def test_resolve_safe_path_empty_path(self, test_config):
        with pytest.raises(ValueError):
            resolve_safe_path("", test_config.workspace_dir)
