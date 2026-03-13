from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService


router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    use_rag: bool = True
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    model_name: str
    tool_traces: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)


@router.post("", summary="Chat")
async def chat(request: ChatRequest):
    if request.stream:
        session_id, model_name, stream = chat_service.stream_chat(
            message=request.message,
            session_id=request.session_id,
            use_rag=request.use_rag,
        )
        return StreamingResponse(
            stream,
            media_type="text/plain; charset=utf-8",
            headers={"X-Session-ID": session_id, "X-Model-Name": model_name},
        )

    result = chat_service.run_chat(
        message=request.message,
        session_id=request.session_id,
        use_rag=request.use_rag,
    )
    return ChatResponse(**result)
