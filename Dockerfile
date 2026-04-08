# Must build FROM openenv-base, not FROM python:3.11-slim
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install CPU-only PyTorch FIRST (prevents the 2GB GPU wheel being pulled
# later by pip when it sees torch>=2.0.0 in requirements)
RUN pip install --no-cache-dir \
    "torch==2.2.2" \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining dependencies (torch is already satisfied above)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the local package (no deps, they're already installed)
RUN pip install --no-cache-dir -e . --no-deps

# Expose the HF Spaces default port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
