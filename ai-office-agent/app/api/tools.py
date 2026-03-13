from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.tool_service import ToolService


router = APIRouter(prefix="/tools", tags=["tools"])
tool_service = ToolService()


class ToolRunRequest(BaseModel):
    tool_name: str
    args: dict = Field(default_factory=dict)


class ToolRunResponse(BaseModel):
    tool_name: str
    output: str


@router.get("", summary="List Tools")
async def list_tools() -> dict[str, list[dict[str, str]]]:
    return {"tools": tool_service.list_available_tools()}


@router.post("/run", response_model=ToolRunResponse, summary="Run Tool")
async def run_tool(request: ToolRunRequest) -> ToolRunResponse:
    result = tool_service.run_tool(
        tool_name=request.tool_name,
        args=request.args,
    )
    return ToolRunResponse(**result)
