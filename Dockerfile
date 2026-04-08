# Must build FROM openenv-base, not FROM python:3.11-slim
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install CPU-only PyTorch first (avoids 2GB GPU wheel download on HF Spaces)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files and install all project deps
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the local package in editable mode
RUN pip install --no-cache-dir -e . --no-deps

# Expose the HF Spaces default port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
