from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage

from app.agent import nodes
from app.core.config import get_settings
from app.llm.factory import LLMFactory


def test_chat_returns_response_shape(client):
    response = client.post(
        "/chat",
        json={
            "message": "hello",
            "use_rag": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["answer"], str)
    assert data["answer"]
    assert "model answer:" in data["answer"]
    assert isinstance(data["model_name"], str)
    assert data["model_name"]
    assert isinstance(data["session_id"], str)
    assert isinstance(data["tool_traces"], list)
    assert isinstance(data["citations"], list)


def test_chat_can_use_draft_email_tool(client):
    response = client.post(
        "/chat",
        json={
            "message": "please draft an email to my manager about project delay",
            "use_rag": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "model tool answer:" in data["answer"]
    assert data["tool_traces"]
    assert data["tool_traces"][0]["tool_name"] == "draft_email"
    assert data["tool_traces"][0]["input"]["purpose"] == "please draft an email to my manager about project delay"


def test_chat_supports_multi_turn(client):
    first_response = client.post(
        "/chat",
        json={
            "message": "My project is Alpha.",
            "use_rag": False,
        },
    )
    assert first_response.status_code == 200
    session_id = first_response.json()["session_id"]

    second_response = client.post(
        "/chat",
        json={
            "message": "What is my project?",
            "session_id": session_id,
            "use_rag": False,
        },
    )

    assert second_response.status_code == 200
    data = second_response.json()
    assert "model multi-turn answer:" in data["answer"]
    assert "My project is Alpha." in data["answer"]


def test_chat_supports_multi_step_tool_loop(client, sample_docs_dir):
    client.post("/rag/ingest", json={"directory": str(sample_docs_dir)})

    response = client.post(
        "/chat",
        json={
            "message": "please draft an email and also search the document for Project Alpha deadline",
            "use_rag": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["tool_traces"]) == 2
    assert data["tool_traces"][0]["tool_name"] == "draft_email"
    assert data["tool_traces"][1]["tool_name"] == "retrieve_docs"
    assert "model tool answer:" in data["answer"]


def test_chat_streams_final_response(client):
    with client.stream(
        "POST",
        "/chat",
        json={
            "message": "hello",
            "use_rag": False,
            "stream": True,
        },
    ) as response:
        body = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert response.headers["x-session-id"]
    assert response.headers["x-model-name"]
    assert "modelanswer:hello" in body.replace(" ", "")


def test_chat_can_use_web_search_tool(client):
    response = client.post(
        "/chat",
        json={
            "message": "What is the latest Qwen news today?",
            "use_rag": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tool_traces"]
    assert data["tool_traces"][0]["tool_name"] == "web_search"
    assert "model tool answer:" in data["answer"]
    assert data["citations"]
    assert data["citations"][0]["url"] == "https://example.com/search-result"


def test_system_prompt_includes_dynamic_time(monkeypatch):
    class CaptureChatModel:
        def __init__(self):
            self.seen_messages = []

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.seen_messages = messages
            return AIMessage(content="captured")

    capture_model = CaptureChatModel()
    monkeypatch.setattr(
        LLMFactory,
        "build_chat_model",
        staticmethod(lambda *args, **kwargs: (capture_model, "capture-model")),
    )

    state = {
        "messages": [{"role": "user", "content": "hello"}],
        "llm_messages": [],
    }
    nodes.plan_node(state)

    system_messages = [
        message.content
        for message in capture_model.seen_messages
        if isinstance(message, SystemMessage)
    ]
    assert system_messages
    now_text = datetime.now().strftime("%Y-%m-%d")
    assert "当前时间：" in system_messages[0]
    assert now_text[:4] in system_messages[0]
