from app.core.config import get_settings
from app.llm.factory import LLMFactory
from app.llm.router import TaskRequirements, choose_chat_model_name


def test_router_prefers_turbo_for_fast_low_cost_tasks(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTING_ENABLED", "true")
    get_settings.cache_clear()

    model_name = choose_chat_model_name(
        TaskRequirements(
            prompt_length=120,
            message_count=1,
            tool_iteration_count=0,
            use_rag=False,
            stream=True,
            requires_reasoning=False,
        )
    )

    assert model_name == get_settings().MODEL_TURBO_NAME
    get_settings.cache_clear()


def test_router_prefers_plus_for_balanced_rag_tasks(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTING_ENABLED", "true")
    get_settings.cache_clear()

    model_name = choose_chat_model_name(
        TaskRequirements(
            prompt_length=900,
            message_count=4,
            tool_iteration_count=1,
            use_rag=True,
            stream=False,
            requires_reasoning=False,
        )
    )

    assert model_name == get_settings().MODEL_PLUS_NAME
    get_settings.cache_clear()


def test_router_prefers_max_for_complex_reasoning_tasks(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTING_ENABLED", "true")
    get_settings.cache_clear()

    model_name = choose_chat_model_name(
        TaskRequirements(
            prompt_length=3200,
            message_count=12,
            tool_iteration_count=2,
            use_rag=True,
            stream=False,
            requires_reasoning=True,
        )
    )

    assert model_name == get_settings().MODEL_MAX_NAME
    get_settings.cache_clear()


def test_factory_can_disable_routing(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTING_ENABLED", "false")
    get_settings.cache_clear()

    model_name = LLMFactory.resolve_chat_model_name(
        task_requirements=TaskRequirements(
            prompt_length=3200,
            message_count=12,
            tool_iteration_count=2,
            use_rag=True,
            stream=False,
            requires_reasoning=True,
        )
    )

    assert model_name == get_settings().MODEL_NAME
    get_settings.cache_clear()
