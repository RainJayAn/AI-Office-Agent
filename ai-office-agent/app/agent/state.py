from typing import TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict]
    llm_messages: list
    response_messages: list
    pending_tool_calls: list[dict]
    user_input: str
    session_id: str
    current_plan: str
    draft_answer: str
    tool_results: list[dict]
    tool_traces: list[dict]
    retrieved_docs: list[dict]
    citations: list[dict]
    final_answer: str
    model_name: str
    need_tool: bool
    selected_tool: str | None
    stream: bool
    use_rag: bool
    iteration_count: int
