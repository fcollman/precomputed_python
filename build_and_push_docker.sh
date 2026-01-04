#!/bin/bash
# Build and push Docker image for precomputed_python with Dask/GCP support
# Builds for Linux (amd64) even on macOS

set -e  # Exit on error

# Configuration
IMAGE_NAME="caveconnectome/precomputed-python-dask"
TAG="${1:-latest}"  # Use first argument as tag, or default to "latest"
PLATFORM="linux/amd64"  # GCP typically uses x86_64

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo "Platform: ${PLATFORM}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if logged in to Docker Hub (optional check - will fail later if not logged in)
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo "Note: Make sure you're logged in to Docker Hub:"
    echo "  docker login"
    echo ""
fi

# Set up buildx for cross-platform builds (if not already set up)
if ! docker buildx ls | grep -q multiarch; then
    echo "Setting up buildx builder for multi-platform builds..."
    docker buildx create --name multiarch --use || true
    docker buildx use multiarch
fi

# Ensure we're using the buildx builder
docker buildx use multiarch 2>/dev/null || docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Build for linux/amd64 platform
echo "Building image..."
docker buildx build \
    --platform "${PLATFORM}" \
    --tag "${IMAGE_NAME}:${TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    --push \
    --file Dockerfile \
    .

echo ""
echo "Successfully built and pushed: ${IMAGE_NAME}:${TAG}"
echo "Successfully built and pushed: ${IMAGE_NAME}:latest"
echo ""
echo "To use this image in your GCP script, use:"
echo "  --gcp-docker-image ${IMAGE_NAME}:${TAG}"

