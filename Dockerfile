FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

# Install dependencies only; mount the repository at runtime.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock /opt/project/
RUN uv sync --locked --no-dev --project /opt/project
ENV PATH="/opt/venv/bin:$PATH"
