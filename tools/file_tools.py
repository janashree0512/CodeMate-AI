"""
CodeMate AI - Sandboxed File Management Tools
=============================================

This module provides file operations strictly confined to the sandboxed 'workspace/'
directory. It prevents path traversal vulnerabilities, rejects absolute paths outside
the workspace, enforces approval gates for destructive operations (overwrites and deletes),
and returns structured tool response dictionaries.

Security Model:
- Path Jail: All operations resolve paths against Config.workspace_dir.
- Traversal Guard: Prevents directory escapes such as '../../'.
- Confirmation Guard: Overwriting or deleting existing files requires explicit approval.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from config import Config, load_config


def resolve_safe_path(target_path: str, workspace_dir: Path) -> Path:
    """
    Resolve and validate that target_path resides strictly inside workspace_dir.
    
    Raises:
        PermissionError: If target_path resolves outside workspace_dir (path traversal).
        ValueError: If target_path is empty.
    """
    if not target_path or not str(target_path).strip():
        raise ValueError("File path cannot be empty.")

    clean_target = Path(target_path.strip())
    resolved_workspace = workspace_dir.resolve()

    if clean_target.is_absolute():
        resolved_target = clean_target.resolve()
    else:
        resolved_target = (resolved_workspace / clean_target).resolve()

    # Enforce directory jail: resolved_target must be relative to resolved_workspace
    try:
        resolved_target.relative_to(resolved_workspace)
    except ValueError as exc:
        raise PermissionError(
            f"Path traversal detected: Path '{target_path}' resolves outside "
            f"sandboxed workspace '{resolved_workspace}'."
        ) from exc

    return resolved_target


def create_file(
    filename: str,
    content: str,
    overwrite: bool = False,
    approved: bool = False,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Create a text file in the sandboxed workspace.

    Args:
        filename: Relative path of the file to create inside workspace.
        content: Text content to write.
        overwrite: Set to True if intending to replace an existing file.
        approved: User approval flag required when overwrite=True on an existing file.
        config: Optional configuration instance (defaults to load_config()).

    Returns:
        Structured response dictionary.
    """
    cfg = config or load_config(strict_api_key=False)
    try:
        target = resolve_safe_path(filename, cfg.workspace_dir)

        # Check if file already exists
        if target.exists():
            if not overwrite:
                return {
                    "success": False,
                    "filename": filename,
                    "error": (
                        f"File '{filename}' already exists. "
                        "Set overwrite=True and obtain user approval to replace it."
                    ),
                    "approval_required": False
                }
            if not approved:
                return {
                    "success": False,
                    "filename": filename,
                    "error": f"Overwriting existing file '{filename}' requires explicit user approval.",
                    "approval_required": True
                }

        # Ensure parent directories exist
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write file contents with UTF-8 encoding
        target.write_text(content, encoding="utf-8")
        bytes_written = len(content.encode("utf-8"))

        return {
            "success": True,
            "filename": filename,
            "bytes_written": bytes_written,
            "message": f"File '{filename}' created successfully ({bytes_written} bytes)."
        }

    except (PermissionError, ValueError) as err:
        return {"success": False, "filename": filename, "error": str(err)}
    except OSError as err:
        return {"success": False, "filename": filename, "error": f"OS error writing file: {err}"}


def read_file(
    filename: str,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Read the contents of a text file inside the sandboxed workspace.

    Args:
        filename: Relative path of the file inside workspace.
        config: Optional configuration instance.

    Returns:
        Structured response dictionary containing content and metadata.
    """
    cfg = config or load_config(strict_api_key=False)
    try:
        target = resolve_safe_path(filename, cfg.workspace_dir)

        if not target.exists():
            return {
                "success": False,
                "filename": filename,
                "error": f"File not found: '{filename}'"
            }

        if target.is_dir():
            return {
                "success": False,
                "filename": filename,
                "error": f"Path '{filename}' is a directory, not a readable file."
            }

        content = target.read_text(encoding="utf-8")
        lines = content.splitlines()

        return {
            "success": True,
            "filename": filename,
            "content": content,
            "line_count": len(lines),
            "size_bytes": len(content.encode("utf-8"))
        }

    except (PermissionError, ValueError) as err:
        return {"success": False, "filename": filename, "error": str(err)}
    except OSError as err:
        return {"success": False, "filename": filename, "error": f"OS error reading file: {err}"}


def list_files(
    directory: str = "",
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    List files and directories inside the sandboxed workspace.

    Args:
        directory: Relative subfolder inside workspace (empty string for root).
        config: Optional configuration instance.

    Returns:
        Structured response dictionary with list of entries.
    """
    cfg = config or load_config(strict_api_key=False)
    try:
        target_dir = resolve_safe_path(directory if directory else ".", cfg.workspace_dir)

        if not target_dir.exists():
            return {
                "success": False,
                "directory": directory,
                "error": f"Directory not found: '{directory}'"
            }

        if not target_dir.is_dir():
            return {
                "success": False,
                "directory": directory,
                "error": f"Path '{directory}' is a file, not a directory."
            }

        entries = []
        for item in sorted(target_dir.iterdir()):
            # Relative path from workspace root
            rel_item = item.relative_to(cfg.workspace_dir.resolve())
            is_file = item.is_file()
            size = item.stat().st_size if is_file else 0

            entries.append({
                "name": item.name,
                "path": str(rel_item).replace("\\", "/"),
                "is_dir": item.is_dir(),
                "size_bytes": size
            })

        return {
            "success": True,
            "directory": directory,
            "total_items": len(entries),
            "entries": entries
        }

    except (PermissionError, ValueError) as err:
        return {"success": False, "directory": directory, "error": str(err)}
    except OSError as err:
        return {"success": False, "directory": directory, "error": f"OS error listing files: {err}"}


def delete_file(
    filename: str,
    approved: bool = False,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Delete a file from the sandboxed workspace with mandatory user approval.

    Args:
        filename: Relative path of the file inside workspace.
        approved: Explicit user approval flag. If False, deletion is blocked.
        config: Optional configuration instance.

    Returns:
        Structured response dictionary.
    """
    cfg = config or load_config(strict_api_key=False)
    try:
        target = resolve_safe_path(filename, cfg.workspace_dir)

        if not target.exists():
            return {
                "success": False,
                "filename": filename,
                "error": f"Cannot delete nonexistent file: '{filename}'"
            }

        if target.is_dir():
            return {
                "success": False,
                "filename": filename,
                "error": f"Cannot delete directory '{filename}' using delete_file."
            }

        # Enforce human-in-the-loop approval gate
        if not approved:
            return {
                "success": False,
                "filename": filename,
                "error": f"Deleting '{filename}' is a destructive action and requires explicit user approval.",
                "approval_required": True
            }

        target.unlink()

        return {
            "success": True,
            "filename": filename,
            "message": f"File '{filename}' successfully deleted."
        }

    except (PermissionError, ValueError) as err:
        return {"success": False, "filename": filename, "error": str(err)}
    except OSError as err:
        return {"success": False, "filename": filename, "error": f"OS error deleting file: {err}"}
