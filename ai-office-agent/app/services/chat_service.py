from collections.abc import Iterator
from uuid import uuid4

from app.agent.nodes import stream_final_response
from app.agent.graph import get_agent_app
from app.agent.state import AgentState
from app.core.config import get_settings


class ChatService:
    def __init__(self) -> None:
        self.agent_app = get_agent_app()
        self.sessions: dict[str, list[dict[str, str]]] = {}

    def run_chat(
        self,
        message: str,
        session_id: str | None = None,
        use_rag: bool = True,
    ) -> dict:
        initial_state = self._build_initial_state(
            message=message,
            session_id=session_id,
            use_rag=use_rag,
            stream=False,
        )
        result = self.agent_app.invoke(initial_state)
        self._save_assistant_message(
            session_id=result.get("session_id", initial_state["session_id"]),
            history=initial_state["messages"],
            answer=result.get("final_answer", ""),
        )

        return {
            "answer": result.get("final_answer", ""),
            "session_id": result.get("session_id", initial_state["session_id"]),
            "model_name": result.get("model_name", ""),
            "tool_traces": result.get("tool_traces", []),
            "citations": result.get("citations", []),
        }

    def stream_chat(
        self,
        message: str,
        session_id: str | None = None,
        use_rag: bool = True,
    ) -> tuple[str, str, Iterator[str]]:
        initial_state = self._build_initial_state(
            message=message,
            session_id=session_id,
            use_rag=use_rag,
            stream=True,
        )
        result = self.agent_app.invoke(initial_state)
        current_session_id = result.get("session_id", initial_state["session_id"])
        response_messages = result.get("response_messages", [])
        model_name = result.get("model_name", "")

        if result.get("final_answer"):
            answer = result["final_answer"]

            def single_chunk_stream() -> Iterator[str]:
                self._save_assistant_message(
                    session_id=current_session_id,
                    history=initial_state["messages"],
                    answer=answer,
                )
                yield answer

            return current_session_id, model_name, single_chunk_stream()

        def answer_stream() -> Iterator[str]:
            chunks: list[str] = []
            for chunk in stream_final_response(response_messages):
                chunks.append(chunk)
                yield chunk
            full_answer = "".join(chunks).strip()
            self._save_assistant_message(
                session_id=current_session_id,
                history=initial_state["messages"],
                answer=full_answer,
            )

        return current_session_id, model_name, answer_stream()

    def _build_initial_state(
        self,
        *,
        message: str,
        session_id: str | None,
        use_rag: bool,
        stream: bool,
    ) -> AgentState:
        settings = get_settings()
        current_session_id = session_id or str(uuid4())
        history = list(self.sessions.get(current_session_id, []))
        history.append({"role": "user", "content": message})
        limited_history = history[-settings.CHAT_HISTORY_LIMIT :]
        return {
            "messages": limited_history,
            "llm_messages": [],
            "response_messages": [],
            "pending_tool_calls": [],
            "user_input": message,
            "session_id": current_session_id,
            "current_plan": "",
            "draft_answer": "",
            "tool_results": [],
            "tool_traces": [],
            "retrieved_docs": [],
            "citations": [],
            "final_answer": "",
            "model_name": "",
            "need_tool": False,
            "selected_tool": None,
            "stream": stream,
            "use_rag": use_rag,
            "iteration_count": 0,
        }

    def _save_assistant_message(
        self,
        *,
        session_id: str,
        history: list[dict[str, str]],
        answer: str,
    ) -> None:
        settings = get_settings()
        assistant_message = {
            "role": "assistant",
            "content": answer,
        }
        self.sessions[session_id] = (list(history) + [assistant_message])[
            -settings.CHAT_HISTORY_LIMIT :
        ]
