"""Errors safe to show in a terminal without leaking credentials."""


class DeepSeekError(Exception):
    """Base error raised by the agent."""


class AuthenticationError(DeepSeekError):
    """The API key was rejected."""


class RateLimitError(DeepSeekError):
    """The API is rate limiting requests after retries."""


class APIError(DeepSeekError):
    """The API returned an unexpected response."""


class ProtocolError(DeepSeekError):
    """The server response did not match the documented protocol."""
