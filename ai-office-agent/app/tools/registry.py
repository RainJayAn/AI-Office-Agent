from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from app.tools.builtins.draft_email import build_draft_email_tool
from app.tools.builtins.retrieve_docs import build_retrieve_docs_tool
from app.tools.builtins.web_search import build_web_search_tool


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    func: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(self, tool: RegisteredTool | dict[str, Any]) -> None:
        if isinstance(tool, dict):
            normalized_tool = RegisteredTool(
                name=tool["name"],
                description=tool.get("description", ""),
                func=tool["func"],
            )
        else:
            normalized_tool = tool

        self._tools[normalized_tool.name] = normalized_tool

    def get_tool(self, tool_name: str) -> RegisteredTool | None:
        return self._tools.get(tool_name)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self._tools.values()
        ]

    def get_langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=tool.func,
                name=tool.name,
                description=tool.description,
            )
            for tool in self._tools.values()
        ]


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(build_draft_email_tool())
    registry.register_tool(build_retrieve_docs_tool())
    registry.register_tool(build_web_search_tool())
    return registry
