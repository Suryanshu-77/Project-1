"""OpenAI-shaped error responses for everything the gateway can raise."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tierwise.providers.base import AuthError, FatalError, ProviderError, RetryableError


class GatewayError(Exception):
    def __init__(
        self, status: int, code: str, message: str, type_: str = "invalid_request_error"
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.type = type_


class InvalidApiKey(GatewayError):
    def __init__(self) -> None:
        super().__init__(401, "invalid_api_key", "Invalid or missing API key.")


class RateLimited(GatewayError):
    def __init__(self, retry_after: int) -> None:
        message = f"Rate limit exceeded for this API key; retry in {retry_after} s."
        super().__init__(429, "rate_limited", message, "rate_limit_error")
        self.retry_after = retry_after


class BudgetExceeded(GatewayError):
    def __init__(self, message: str) -> None:
        super().__init__(429, "budget_exceeded", message, "insufficient_quota")


class AllCircuitsOpen(GatewayError):
    """Every candidate's circuit breaker is open, so nothing was attempted."""

    def __init__(self, skipped: list[str]) -> None:
        names = ", ".join(skipped) or "none"
        message = f"All candidate models are temporarily unavailable (circuit open: {names})."
        super().__init__(503, "upstream_unavailable", message, "api_error")


def error_response(status: int, code: str, message: str, type_: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": type_, "code": code}}
    )


def gateway_error_from(exc: ProviderError) -> GatewayError:
    if isinstance(exc, AuthError):
        # Never echo the upstream body: it can contain (part of) our key.
        message = f"Upstream provider '{exc.provider}' rejected the gateway's credentials."
        return GatewayError(502, "upstream_auth_error", message, "api_error")
    if isinstance(exc, FatalError):
        return GatewayError(exc.status or 400, "invalid_request_error", exc.message)
    if isinstance(exc, RetryableError):
        message = f"Upstream provider '{exc.provider}' is unavailable: {exc.message}"
        return GatewayError(503, "upstream_unavailable", message, "api_error")
    return GatewayError(502, "upstream_error", exc.message, "api_error")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def _gateway(_: Request, exc: GatewayError) -> JSONResponse:
        response = error_response(exc.status, exc.code, exc.message, exc.type)
        if isinstance(exc, RateLimited):
            response.headers["Retry-After"] = str(exc.retry_after)
        return response

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
        err = gateway_error_from(exc)
        return error_response(err.status, err.code, err.message, err.type)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return error_response(400, "invalid_request_error", problems, "invalid_request_error")
