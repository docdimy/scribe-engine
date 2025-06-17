#!/bin/bash

# 🏥 Scribe Engine Setup Script
# Easy deployment of the microservice

set -e

echo "🏥 Scribe Engine setup starting..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker found"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_status "Docker Compose found"

# Check/create .env file
echo "📝 Preparing configuration..."

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        print_warning ".env file created from .env.example"
        print_warning "⚠️  IMPORTANT: Please edit .env and set your API keys!"
        print_warning "   - OPENAI_API_KEY"
        print_warning "   - ASSEMBLYAI_API_KEY"
        print_warning "   - API_SECRET_KEY"
        echo ""
        read -p "Press Enter when you have configured the .env file..."
    else
        print_error ".env.example not found"
        exit 1
    fi
else
    print_status ".env file found"
fi

# Validate API keys
echo "🔑 Checking API keys..."

source .env

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key" ]; then
    print_error "OPENAI_API_KEY not set. Please configure .env"
    exit 1
fi
print_status "OpenAI API key set"

if [ -z "$ASSEMBLYAI_API_KEY" ] || [ "$ASSEMBLYAI_API_KEY" = "your-assemblyai-api-key" ]; then
    print_error "ASSEMBLYAI_API_KEY not set. Please configure .env"
    exit 1
fi
print_status "AssemblyAI API key set"

if [ -z "$API_SECRET_KEY" ] || [ "$API_SECRET_KEY" = "your-secret-key-here" ]; then
    print_warning "API_SECRET_KEY should be changed for production"
fi

# Start Docker Compose
echo "🚀 Starting services..."

# Stop old containers if present
docker-compose down 2>/dev/null || true

# Build images and start services
echo "📦 Building Docker images..."
docker-compose build

echo "🔄 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for service startup..."
sleep 10

# Health check
echo "🏥 Performing health check..."

max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f -s http://localhost:3001/health > /dev/null; then
        print_status "Scribe Engine is ready!"
        break
    else
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    print_error "Service could not be started"
    echo "📋 Show logs:"
    docker-compose logs
    exit 1
fi

# Successful installation
echo ""
echo "🎉 Scribe Engine successfully installed!"
echo ""
echo "📋 Available services:"
echo "   • Scribe Engine API: http://localhost:3001"
echo "   • Health Check:      http://localhost:3001/health"
echo "   • API Documentation: http://localhost:3001/docs"
echo "   • Prometheus:        http://localhost:9090"
echo "   • Redis:             localhost:6379"
echo ""
echo "🔍 Testing:"
echo "   curl http://localhost:3001/health"
echo ""
echo "📚 More information:"
echo "   • README.md for complete documentation"
echo "   • docker-compose logs -f for live logs"
echo "   • docker-compose down to stop"
echo ""
print_status "Setup completed!" 