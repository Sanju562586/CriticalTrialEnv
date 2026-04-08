# Multi-stage build using openenv-base (matches official OpenEnv pattern)
# Reference: github.com/meta-pytorch/OpenEnv/envs/echo_env/server/Dockerfile

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

ARG BUILD_MODE=standalone

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv > /dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies using uv sync (no lockfile - generate fresh)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# --- Runtime stage ---
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment and code from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Use venv Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

EXPOSE 7860

# Start the FastAPI server on port 7860 (HF Spaces requirement)
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
