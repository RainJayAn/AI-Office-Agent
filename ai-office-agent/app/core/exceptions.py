class AppBaseException(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_code: str = "app_error",
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class ToolExecutionError(AppBaseException):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        details: dict | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            error_code="tool_execution_error",
            details=details,
        )


class LLMProviderError(AppBaseException):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        details: dict | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            error_code="llm_provider_error",
            details=details,
        )


class RAGPipelineError(AppBaseException):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        details: dict | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            error_code="rag_pipeline_error",
            details=details,
        )
