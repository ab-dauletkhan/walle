"""
Base Tool Executor class for WALL-E
Provides common execute() pattern for all tool executors.
"""

from abc import ABC
from typing import Dict, List


class BaseToolExecutor(ABC):
    """
    Abstract base class for tool executors.

    Subclasses implement methods with naming pattern: {method_prefix}{tool_name}
    Default prefix is "_", override method_prefix for different patterns.
    """

    method_prefix: str = "_"

    def execute(self, fn_name: str, args: dict) -> str:
        """
        Execute a tool by name with given arguments.

        Args:
            fn_name: Name of the tool/function to execute
            args: Dictionary of arguments to pass to the tool

        Returns:
            String result of the tool execution
        """
        method = getattr(self, f"{self.method_prefix}{fn_name}", None)
        if method:
            return method(args)
        return f"Unknown tool: {fn_name}"

    def get_tool_schemas(self) -> List[Dict]:
        """
        Return tool schemas in OpenAI format.
        Subclasses should override this method.
        """
        return []
