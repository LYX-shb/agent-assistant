"""Errors that should be returned as a temporary service-unavailable response."""


class ServiceUnavailableError(RuntimeError):
    """Raised when an upstream model, retriever, or knowledge base is unavailable."""
