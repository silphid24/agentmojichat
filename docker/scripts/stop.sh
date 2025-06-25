#!/bin/bash

echo "Stopping MOJI AI Agent development environment..."

# Stop services
docker-compose -f docker-compose.dev.yml down

echo "All services stopped."