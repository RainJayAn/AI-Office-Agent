from langchain_openai import ChatOpenAI


def get_openai_compatible_chat_model(
    api_key: str | None,
    base_url: str,
    model_name: str,
    temperature: float = 0.2,
    timeout: int = 60,
    max_retries: int = 2,
):
    if not api_key:
        return None

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
