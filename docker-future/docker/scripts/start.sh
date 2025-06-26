#!/bin/bash

echo "Starting MOJI AI Agent development environment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
fi

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 5

# Check service status
docker-compose -f docker-compose.dev.yml ps

echo "MOJI AI Agent is running at http://localhost:8000"
echo "PostgreSQL is available at localhost:5432"
echo "Redis is available at localhost:6379"
echo ""
echo "To view logs: docker-compose -f docker-compose.dev.yml logs -f"
echo "To stop: ./docker/scripts/stop.sh"