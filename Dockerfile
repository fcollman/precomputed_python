# Dockerfile for precomputed_python with Dask and GCP dependencies
# Compatible with dask-cloudprovider for running distributed annotation pipelines on GCP

FROM daskdev/dask:latest

# Set working directory
WORKDIR /app

# Install system dependencies if needed
# (daskdev/dask:latest typically has most build tools, but we ensure we have them)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency files first (for better layer caching)
# Only copy files needed for dependency installation
# pyproject.toml is required, README.md is referenced by package metadata
COPY pyproject.toml README.md /app/precomputed_python/

# Set working directory to precomputed_python source
WORKDIR /app/precomputed_python

# Install dependencies from pyproject.toml using a Python script
# Extract and install dependencies dynamically from pyproject.toml
# This ensures we use the exact versions specified in the file
RUN pip install --no-cache-dir tomli && \
    python3 << 'EOF'
import subprocess
import sys

# Use tomli (works on all Python versions, daskdev/dask may have Python < 3.11)
import tomli

# Read pyproject.toml
with open('pyproject.toml', 'rb') as f:
    data = tomli.load(f)

# Extract base dependencies
base_deps = data.get('project', {}).get('dependencies', [])

# Extract distributed-annotations-gcp group dependencies
dep_groups = data.get('dependency-groups', {})
gcp_deps = dep_groups.get('distributed-annotations-gcp', [])

# Add pyarrow (needed for dask.dataframe but not in dependency groups)
all_deps = base_deps + gcp_deps + ['pyarrow>=10.0.0']

# Install all dependencies
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + all_deps)
EOF

# Now copy the source code (this layer will be rebuilt when source changes)
COPY src/ /app/precomputed_python/src/
COPY examples/ /app/precomputed_python/examples/

# Install precomputed_python package in editable mode
# This is fast since dependencies are already installed
RUN pip install --no-cache-dir -e .

# Verify installation
RUN python -c "import precomputed_python; import dask; import dask_cloudprovider; import pyarrow; import tensorstore; import neuroglancer; print('All dependencies installed successfully')"

# Set working directory back to /app for running scripts
WORKDIR /app

# Default command (dask scheduler will be started separately by dask-cloudprovider)
CMD ["/bin/bash"]

