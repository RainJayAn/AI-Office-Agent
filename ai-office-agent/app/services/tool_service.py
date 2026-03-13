from app.core.exceptions import ToolExecutionError
from app.tools.registry import get_tool_registry


class ToolService:
    def __init__(self) -> None:
        self.registry = get_tool_registry()

    def list_available_tools(self) -> list[dict[str, str]]:
        return self.registry.list_tools()

    def run_tool(self, tool_name: str, args: dict) -> dict:
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            raise ToolExecutionError(
                f"Tool not found: {tool_name}",
                status_code=404,
                details={"tool_name": tool_name},
            )

        try:
            result = tool.func(**args)
        except TypeError as exc:
            raise ToolExecutionError(
                str(exc),
                status_code=400,
                details={"tool_name": tool_name, "args": args},
            ) from exc
        except Exception as exc:
            raise ToolExecutionError(
                f"Failed to run tool: {tool_name}",
                status_code=500,
                details={"tool_name": tool_name},
            ) from exc

        return {
            "tool_name": tool.name,
            "output": result,
        }
