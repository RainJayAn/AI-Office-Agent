import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.path import clear_path_caches
from app.llm.factory import LLMFactory
from app.main import app
from app.rag import retriever as rag_retriever
from app.tools.builtins import web_search as web_search_tool


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        normalized = text.lower()
        features = [
            float(normalized.count("project")),
            float(normalized.count("alpha")),
            float(normalized.count("deadline")),
            float(normalized.count("budget")),
            float(len(normalized)),
        ]
        return features

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class FakeDDGS:
    def text(self, query: str, max_results: int = 5):
        return [
            {
                "title": f"Search result for {query}",
                "href": "https://example.com/search-result",
                "body": f"Latest information for query: {query}",
            }
        ][:max_results]

    def close(self) -> None:
        return None


class FakeChatModel:
    def __init__(self, use_tools: bool = False) -> None:
        self.use_tools = use_tools

    def bind_tools(self, tools: list[BaseTool]):
        return FakeChatModel(use_tools=True)

    def invoke(self, messages):
        human_messages = [
            message.content
            for message in messages
            if isinstance(message, HumanMessage)
        ]
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        latest_user = human_messages[-1]

        if self.use_tools:
            lowered = latest_user.lower()
            used_tool_names = {message.name for message in tool_messages}

            if any(keyword in lowered for keyword in ("latest", "today", "news", "current", "recent")):
                if "web_search" not in used_tool_names:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "web_search",
                                "args": {
                                    "query": latest_user,
                                    "max_results": 3,
                                },
                                "id": "call_web_search",
                            }
                        ],
                    )

            if "email" in lowered or "mail" in lowered:
                if "document" in lowered and "draft_email" not in used_tool_names:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "draft_email",
                                "args": {
                                    "recipient": "待确认",
                                    "subject": "待确认主题",
                                    "purpose": latest_user,
                                    "key_points": [],
                                },
                                "id": "call_draft_email",
                            }
                        ],
                    )
                if "document" in lowered and "retrieve_docs" not in used_tool_names:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "retrieve_docs",
                                "args": {
                                    "query": latest_user,
                                    "top_k": 3,
                                },
                                "id": "call_retrieve_docs_after_email",
                            }
                        ],
                    )
                if "draft_email" not in used_tool_names:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "draft_email",
                                "args": {
                                    "recipient": "待确认",
                                    "subject": "待确认主题",
                                    "purpose": latest_user,
                                    "key_points": [],
                                },
                                "id": "call_draft_email",
                            }
                        ],
                    )

            if ("document" in lowered or "knowledge base" in lowered) and "retrieve_docs" not in used_tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_docs",
                            "args": {
                                "query": latest_user,
                                "top_k": 3,
                            },
                            "id": "call_retrieve_docs",
                        }
                    ],
                )

        if tool_messages:
            joined_tool_outputs = " | ".join(message.content for message in tool_messages)
            return AIMessage(content=f"model tool answer: {joined_tool_outputs}")
        if len(human_messages) >= 2:
            return AIMessage(
                content=(
                    "model multi-turn answer: "
                    f"previous={human_messages[-2]} | current={human_messages[-1]}"
                )
            )
        return AIMessage(content=f"model answer: {latest_user}")

    def stream(self, messages):
        full_text = self.invoke(messages).content
        for token in full_text.split(" "):
            yield AIMessage(content=f"{token} ")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("VECTOR_DB_PATH", "rag_store")
    monkeypatch.setenv("APP_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("APP_USER_DATA_DIR", str(tmp_path / "user_data"))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setattr(
        LLMFactory,
        "build_chat_model",
        staticmethod(lambda *args, **kwargs: (FakeChatModel(), "fake-model")),
    )
    monkeypatch.setattr(rag_retriever, "get_embedding_model", lambda: FakeEmbeddings())
    monkeypatch.setattr(web_search_tool, "_create_ddgs", lambda: FakeDDGS())
    clear_path_caches()
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    clear_path_caches()
    get_settings.cache_clear()


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "office.md").write_text(
        "# Office Guide\n"
        "Project Alpha deadline is March 20.\n"
        "The review meeting is every Friday afternoon.\n",
        encoding="utf-8",
    )
    (docs_dir / "notes.txt").write_text(
        "Budget approval owner is the finance team.\n",
        encoding="utf-8",
    )
    return docs_dir
