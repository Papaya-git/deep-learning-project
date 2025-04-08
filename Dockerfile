# Runtime stage
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /app

# # Install system dependencies first
# RUN apt-get update && apt-get install -y \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Use PIP cache directory
ENV PIP_CACHE_DIR=/app/.pip-cache
RUN mkdir -p ${PIP_CACHE_DIR}

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY *.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TF_FORCE_GPU_ALLOW_GROWTH=true


# Default command (override in compose)
CMD ["python", "src/main.py"]