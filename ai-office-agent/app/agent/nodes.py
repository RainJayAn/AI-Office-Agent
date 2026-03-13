from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.state import AgentState
from app.core.config import get_settings
from app.core.exceptions import LLMProviderError, ToolExecutionError
from app.llm.factory import LLMFactory
from app.llm.router import TaskRequirements
from app.prompts import AGENT_REASONING_KEYWORDS, build_agent_system_prompt
from app.tools.registry import get_tool_registry


def plan_node(state: AgentState) -> AgentState:
    llm_messages = _get_working_messages(state)
    ai_message, _ = _invoke_model(llm_messages, use_tools=True)
    pending_tool_calls = list(getattr(ai_message, "tool_calls", []) or [])
    draft_answer = ""
    iteration_count = state.get("iteration_count", 0) + 1
    max_iterations = get_settings().TOOL_CALL_MAX_ITERATIONS

    if pending_tool_calls:
        llm_messages.append(ai_message)
        if iteration_count >= max_iterations:
            draft_answer = "工具调用次数已达到上限，请基于现有信息直接回答。"
            pending_tool_calls = []

    return {
        **state,
        "llm_messages": llm_messages,
        "pending_tool_calls": pending_tool_calls,
        "draft_answer": draft_answer,
        "need_tool": bool(pending_tool_calls),
        "current_plan": "tool_call" if pending_tool_calls else "respond_directly",
        "iteration_count": iteration_count,
    }


def tool_node(state: AgentState) -> AgentState:
    registry = get_tool_registry()
    llm_messages = list(state.get("llm_messages", []))
    tool_results = list(state.get("tool_results", []))
    tool_traces = list(state.get("tool_traces", []))
    citations = list(state.get("citations", []))
    retrieved_docs = list(state.get("retrieved_docs", []))

    for tool_call in state.get("pending_tool_calls", []):
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {}) or {}
        tool_call_id = tool_call.get("id", tool_name)
        tool = registry.get_tool(tool_name)
        if tool is None:
            raise ToolExecutionError(
                f"Tool not found: {tool_name}",
                status_code=404,
                details={"tool_name": tool_name},
            )

        try:
            output = tool.func(**tool_args)
        except TypeError as exc:
            raise ToolExecutionError(
                str(exc),
                status_code=400,
                details={"tool_name": tool_name, "args": tool_args},
            ) from exc
        except Exception as exc:
            raise ToolExecutionError(
                f"Failed to run tool: {tool_name}",
                status_code=500,
                details={"tool_name": tool_name},
            ) from exc

        tool_results.append({"tool_name": tool_name, "output": output})
        tool_traces.append(
            {
                "tool_name": tool_name,
                "status": "success",
                "input": tool_args,
                "output_preview": str(output)[:120],
            }
        )
        llm_messages.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

        if tool_name == "retrieve_docs":
            retrieved_docs.append(
                {
                    "query": tool_args.get("query", ""),
                    "output": output,
                }
            )
            citations.extend(_extract_retrieve_docs_citations(str(output)))
        elif tool_name == "web_search":
            citations.extend(_extract_web_search_citations(str(output)))

    return {
        **state,
        "llm_messages": llm_messages,
        "tool_results": tool_results,
        "tool_traces": tool_traces,
        "retrieved_docs": retrieved_docs,
        "citations": citations,
        "pending_tool_calls": [],
    }


def respond_node(state: AgentState) -> AgentState:
    response_messages = list(state.get("llm_messages", []))
    response_model_name = resolve_response_model_name(response_messages)

    if state.get("draft_answer"):
        return {
            **state,
            "response_messages": response_messages,
            "final_answer": state.get("draft_answer", ""),
            "model_name": response_model_name,
            "citations": list(state.get("citations", [])),
        }

    if state.get("stream"):
        return {
            **state,
            "response_messages": response_messages,
            "final_answer": "",
            "model_name": response_model_name,
            "citations": list(state.get("citations", [])),
        }

    final_answer, model_name = generate_final_response(response_messages)
    return {
        **state,
        "response_messages": response_messages,
        "final_answer": final_answer,
        "model_name": model_name,
        "citations": list(state.get("citations", [])),
    }


def should_use_tool(state: AgentState) -> str:
    return "tool" if state.get("need_tool") else "respond"


def generate_final_response(messages: list) -> tuple[str, str]:
    final_ai_message, model_name = _invoke_model(
        messages,
        use_tools=False,
        task_requirements=_build_task_requirements_from_messages(
            messages=messages,
            use_rag=any(
                isinstance(message, ToolMessage) and message.name == "retrieve_docs"
                for message in messages
            ),
            stream=False,
        ),
    )
    content = _extract_content(final_ai_message)
    if not content:
        raise LLMProviderError(
            "Model returned an empty final response.",
            details={"provider": "openai-compatible"},
        )
    return content, model_name


def stream_final_response(messages: list) -> Iterator[str]:
    task_requirements = _build_task_requirements_from_messages(
        messages=messages,
        use_rag=any(
            isinstance(message, ToolMessage) and message.name == "retrieve_docs"
            for message in messages
        ),
        stream=True,
    )
    model, _ = LLMFactory.build_chat_model(task_requirements=task_requirements)
    if model is None:
        raise LLMProviderError(
            "Chat model is not configured. Please set DASHSCOPE_API_KEY.",
            status_code=500,
        )

    try:
        for chunk in model.stream(messages):
            text = _extract_content(chunk)
            if text:
                yield text
    except Exception as exc:
        raise LLMProviderError(
            "Failed to stream chat response from the configured model.",
            details={"provider": "openai-compatible"},
        ) from exc


def resolve_response_model_name(messages: list) -> str:
    task_requirements = _build_task_requirements_from_messages(
        messages=messages,
        use_rag=any(
            isinstance(message, ToolMessage) and message.name == "retrieve_docs"
            for message in messages
        ),
        stream=False,
    )
    return LLMFactory.resolve_chat_model_name(task_requirements=task_requirements)


def _get_working_messages(state: AgentState) -> list:
    existing_messages = list(state.get("llm_messages", []))
    if existing_messages:
        return existing_messages

    llm_messages: list = [SystemMessage(content=_build_system_prompt())]

    for message in state.get("messages", []):
        role = message.get("role")
        content = message.get("content", "")
        if not content:
            continue
        if role == "user":
            llm_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            llm_messages.append(AIMessage(content=content))

    return llm_messages


def _build_system_prompt() -> str:
    current_time = _get_current_time()
    current_time_text = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    return build_agent_system_prompt(current_time_text)


def _get_current_time() -> datetime:
    settings = get_settings()
    try:
        return datetime.now(ZoneInfo(settings.APP_TIMEZONE))
    except ZoneInfoNotFoundError:
        if settings.APP_TIMEZONE == "Asia/Shanghai":
            return datetime.now(timezone(timedelta(hours=8), name="CST"))
        return datetime.now().astimezone()


def _invoke_model(
    messages: list,
    *,
    use_tools: bool,
    task_requirements: TaskRequirements | None = None,
) -> tuple[AIMessage, str]:
    effective_task_requirements = task_requirements or _build_task_requirements_from_messages(
        messages=messages,
        use_rag=any(
            isinstance(message, ToolMessage) and message.name == "retrieve_docs"
            for message in messages
        ),
        stream=False,
    )
    model, model_name = LLMFactory.build_chat_model(task_requirements=effective_task_requirements)
    if model is None:
        raise LLMProviderError(
            "Chat model is not configured. Please set DASHSCOPE_API_KEY.",
            status_code=500,
        )

    runnable = model.bind_tools(get_tool_registry().get_langchain_tools()) if use_tools else model

    try:
        response = runnable.invoke(messages)
    except Exception as exc:
        raise LLMProviderError(
            "Failed to generate chat response from the configured model.",
            details={"provider": "openai-compatible", "use_tools": use_tools, "model_name": model_name},
        ) from exc

    if not isinstance(response, AIMessage):
        response = AIMessage(content=_extract_content(response))

    if not _extract_content(response).strip() and not getattr(response, "tool_calls", None):
        raise LLMProviderError(
            "Model returned an empty response.",
            details={"provider": "openai-compatible", "model_name": model_name},
        )
    return response, model_name


def _build_task_requirements_from_messages(
    *,
    messages: list,
    use_rag: bool,
    stream: bool,
) -> TaskRequirements:
    text_parts: list[str] = []
    message_count = 0
    tool_iteration_count = 0
    requires_reasoning = False

    for message in messages:
        content = _extract_content(message)
        if content:
            text_parts.append(content)
        if isinstance(message, (HumanMessage, AIMessage, ToolMessage)):
            message_count += 1
        if isinstance(message, ToolMessage):
            tool_iteration_count += 1
        if isinstance(message, HumanMessage):
            lowered = content.lower()
            if any(keyword in lowered for keyword in AGENT_REASONING_KEYWORDS):
                requires_reasoning = True

    prompt_length = sum(len(part) for part in text_parts)
    return TaskRequirements(
        prompt_length=prompt_length,
        message_count=message_count,
        tool_iteration_count=tool_iteration_count,
        use_rag=use_rag,
        stream=stream,
        requires_reasoning=requires_reasoning,
    )


def _extract_content(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _extract_retrieve_docs_citations(output: str) -> list[dict]:
    citations: list[dict] = []
    current_source: str | None = None
    current_page: str | None = None
    for line in output.splitlines():
        if line.startswith("来源："):
            current_source = line.replace("来源：", "", 1).strip()
        elif line.startswith("页码："):
            current_page = line.replace("页码：", "", 1).strip()
        elif current_source and line.startswith("片段："):
            citation = {
                "source": current_source,
                "snippet": line.replace("片段：", "", 1).strip(),
            }
            if current_page:
                citation["page"] = current_page
            citations.append(citation)
            current_source = None
            current_page = None
    return citations


def _extract_web_search_citations(output: str) -> list[dict]:
    citations: list[dict] = []
    current_query = ""
    current_title = ""
    current_url = ""
    for line in output.splitlines():
        if line.startswith("检索词："):
            current_query = line.replace("检索词：", "", 1).strip()
        elif line.startswith("标题："):
            current_title = line.replace("标题：", "", 1).strip()
        elif line.startswith("链接："):
            current_url = line.replace("链接：", "", 1).strip()
        elif line.startswith("摘要："):
            citations.append(
                {
                    "source": current_title or current_url,
                    "url": current_url,
                    "snippet": line.replace("摘要：", "", 1).strip(),
                    "query": current_query,
                }
            )
            current_query = ""
            current_title = ""
            current_url = ""
    return citations
