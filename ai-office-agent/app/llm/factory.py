from app.core.config import get_settings
from app.llm.providers.openai_compatible import get_openai_compatible_chat_model
from app.llm.router import TaskRequirements, choose_chat_model_name


class LLMFactory:
    @staticmethod
    def build_chat_model(
        provider: str | None = None,
        model_name: str | None = None,
        task_requirements: TaskRequirements | None = None,
    ):
        settings = get_settings()
        selected_provider = (provider or settings.MODEL_PROVIDER).lower()
        selected_model = LLMFactory.resolve_chat_model_name(
            provider=provider,
            model_name=model_name,
            task_requirements=task_requirements,
        )

        if selected_provider in {"dashscope", "openai-compatible", "openai_compatible"}:
            model = get_openai_compatible_chat_model(
                api_key=settings.API_KEY,
                base_url=settings.BASE_URL,
                model_name=selected_model,
                temperature=settings.MODEL_TEMPERATURE,
                timeout=settings.MODEL_TIMEOUT,
                max_retries=settings.MODEL_MAX_RETRIES,
            )
            return model, selected_model

        raise ValueError(f"Unsupported model provider: {selected_provider}")

    @staticmethod
    def resolve_chat_model_name(
        provider: str | None = None,
        model_name: str | None = None,
        task_requirements: TaskRequirements | None = None,
    ) -> str:
        settings = get_settings()
        selected_provider = (provider or settings.MODEL_PROVIDER).lower()
        if selected_provider not in {"dashscope", "openai-compatible", "openai_compatible"}:
            raise ValueError(f"Unsupported model provider: {selected_provider}")

        if model_name:
            return model_name

        if settings.MODEL_ROUTING_ENABLED:
            return choose_chat_model_name(task_requirements)

        return settings.MODEL_NAME

    @staticmethod
    def get_chat_model(
        provider: str | None = None,
        model_name: str | None = None,
        task_requirements: TaskRequirements | None = None,
    ):
        model, _ = LLMFactory.build_chat_model(
            provider=provider,
            model_name=model_name,
            task_requirements=task_requirements,
        )
        return model
