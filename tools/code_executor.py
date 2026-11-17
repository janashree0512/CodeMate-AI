"""
Backward-compatibility alias for tools/execution_tool.py.
"""
from tools.execution_tool import (
    BaseExecutor,
    LocalRestrictedExecutor,
    build_sanitized_environment,
    execute_python,
)

__all__ = [
    "BaseExecutor",
    "LocalRestrictedExecutor",
    "build_sanitized_environment",
    "execute_python",
]
