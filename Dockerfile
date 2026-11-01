FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1

WORKDIR /app
# dependencies first, so source edits don't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY deploy/registry.compose.yaml ./deploy/registry.compose.yaml
COPY models/router ./models/router
COPY scripts/fake_upstream.py ./scripts/fake_upstream.py
RUN uv sync --frozen --no-dev

RUN useradd --system --no-create-home tierwise
USER tierwise

ENV TIERWISE_HOST=0.0.0.0 TIERWISE_PORT=8000
EXPOSE 8000
CMD ["tierwise"]
