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


def error_response(status: int, code: str, message: str, type_: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": type_, "code": code}}
    )


def _from_provider_error(exc: ProviderError) -> GatewayError:
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
        return error_response(exc.status, exc.code, exc.message, exc.type)

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
        err = _from_provider_error(exc)
        return error_response(err.status, err.code, err.message, err.type)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return error_response(400, "invalid_request_error", problems, "invalid_request_error")
