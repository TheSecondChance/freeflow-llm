class FreeFlowError(Exception):
    """Base exception for all FreeFlow LLM errors."""

    pass


class RateLimitError(FreeFlowError):
    """Raised when a provider hits rate limit (HTTP 429)."""

    def __init__(self, provider: str, message: str = "Rate limit exceeded"):
        self.provider = provider
        self.message = f"{provider}: {message}"
        super().__init__(self.message)


class ProviderError(FreeFlowError):
    """Raised when a provider encounters an error."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = f"{provider}: {message}"
        super().__init__(self.message)


class NoProvidersAvailableError(FreeFlowError):
    """Raised when all providers have been exhausted or none are configured."""

    def __init__(self, message: str = "All providers exhausted or none configured"):
        super().__init__(message)


class InvalidAPIKeyError(FreeFlowError):
    """Raised when an API key is invalid or missing."""

    def __init__(self, provider: str, message: str = "Invalid or missing API key"):
        self.provider = provider
        self.message = f"{provider}: {message}"
        super().__init__(self.message)
