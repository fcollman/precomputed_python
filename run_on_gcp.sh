#!/bin/bash
# Script to run synapse_csv_to_annotations_gcp.py on Google Cloud Platform
# 
# Options:
# 1. Run from Google Cloud Shell (easiest) - just run this script there
# 2. Run from a Compute Engine VM - SSH into VM and run this script
#
# Prerequisites:
# - Docker image built and pushed to Docker Hub or GCR
# - gcloud CLI installed and authenticated (gcloud auth login)
# - Appropriate GCP permissions to create Compute Engine instances

set -e

# Configuration - adjust these as needed
DOCKER_IMAGE="${DOCKER_IMAGE:-caveconnectome/precomputed-python-dask:latest}"
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo '')}"
GCP_ZONE="${GCP_ZONE:-us-east1-b}"

# Check if running on GCP (Cloud Shell or VM)
if [ -n "$CLOUD_SHELL" ] || [ -f /sys/class/dmi/id/product_name ] && grep -q "Google" /sys/class/dmi/id/product_name 2>/dev/null; then
    echo "Running on GCP environment - using internal networking"
    USE_INTERNAL_NETWORK=true
else
    echo "WARNING: Not running on GCP. For best practices, run from Cloud Shell or a Compute Engine VM."
    echo "The Dask cluster will use public networking (public_ingress=True)."
    USE_INTERNAL_NETWORK=false
fi

# Set default project if not set
if [ -z "$GCP_PROJECT" ]; then
    echo "Error: GCP project not set. Set GCP_PROJECT environment variable or run:"
    echo "  gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "Using GCP project: $GCP_PROJECT"
echo "Using Docker image: $DOCKER_IMAGE"
echo ""

# Pull the Docker image (if using Docker Hub, or it's already in GCR)
if [[ "$DOCKER_IMAGE" == gcr.io/* ]]; then
    echo "Using GCR image: $DOCKER_IMAGE"
elif [[ "$DOCKER_IMAGE" == caveconnectome/* ]]; then
    echo "Using Docker Hub image: $DOCKER_IMAGE"
    docker pull "$DOCKER_IMAGE"
else
    echo "Using Docker image: $DOCKER_IMAGE"
fi

# Run the script with the Docker image
# The script will handle creating the Dask cluster
python3 examples/synapse_csv_to_annotations_gcp.py \
    --gcp-project "$GCP_PROJECT" \
    --gcp-docker-image "$DOCKER_IMAGE" \
    --gcp-zone "$GCP_ZONE" \
    "$@"

