#!/bin/bash

# Build and Push Script for Private Registry
# Usage: ./scripts/build-and-push.sh [tag]

set -e

# Configuration
REGISTRY_HOST="localhost:5000"
IMAGE_NAME="scribe-engine"
TAG=${1:-latest}
FULL_IMAGE_NAME="${REGISTRY_HOST}/${IMAGE_NAME}:${TAG}"

echo "🏗️  Building Docker image..."
docker build -t ${FULL_IMAGE_NAME} .

echo "📦 Pushing to private registry..."
docker push ${FULL_IMAGE_NAME}

echo "✅ Image pushed successfully!"
echo "   Image: ${FULL_IMAGE_NAME}"
echo "   Registry UI: http://localhost:5001"

echo ""
echo "🚀 To use in docker-compose:"
echo "   image: ${FULL_IMAGE_NAME}" 