"""Run the gateway: ``python -m tierwise``."""

import uvicorn

from tierwise.api.app import create_app
from tierwise.config import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
