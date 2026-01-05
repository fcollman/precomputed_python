# Dockerfile for precomputed_python with Dask and GCP dependencies
# Compatible with dask-cloudprovider for running distributed annotation pipelines on GCP
# Uses Python 3.12 with uv for modern, fast dependency management

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# tini is needed for proper signal handling in containers (required by dask-cloudprovider)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package installer - much faster than pip)
# Copy uv binary from official uv image for better caching
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (for better layer caching)
# Only copy files needed for dependency installation
COPY pyproject.toml README.md /app/precomputed_python/

# Set working directory to precomputed_python source
WORKDIR /app/precomputed_python

# Install dependencies using uv
# uv is much faster than pip and handles dependency resolution better
# First install tomli to parse pyproject.toml, then extract and install dependencies
RUN uv pip install --system --no-cache tomli && \
    python3 -c "import tomli; f=open('pyproject.toml','rb'); d=tomli.load(f); f.close(); b=d.get('project',{}).get('dependencies',[]); g=d.get('dependency-groups',{}).get('distributed-annotations-gcp',[]); a=b+g+['pyarrow>=10.0.0']; print(' '.join(a))" > /tmp/deps.txt && \
    uv pip install --system --no-cache $(cat /tmp/deps.txt)

# Now copy the source code (this layer will be rebuilt when source changes)
COPY src/ /app/precomputed_python/src/
COPY examples/ /app/precomputed_python/examples/

# Install precomputed_python package in editable mode using uv
RUN uv pip install --system --no-cache -e .

# Verify installation
RUN python -c "import precomputed_python; import dask; import dask_cloudprovider; import pyarrow; import tensorstore; import neuroglancer; print('All dependencies installed successfully')"

# Set working directory back to /app for running scripts
WORKDIR /app

# Use tini as entrypoint for proper signal handling (required by dask-cloudprovider)
# dask-cloudprovider will override CMD with dask-scheduler or dask-worker commands
ENTRYPOINT ["tini", "--"]

# Default command (will be overridden by dask-cloudprovider)
CMD ["/bin/bash"]
