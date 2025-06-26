#!/bin/bash

echo "Cleaning MOJI AI Agent development environment..."

# Stop and remove containers, networks, volumes
docker-compose -f docker-compose.dev.yml down -v

# Remove images
docker-compose -f docker-compose.dev.yml down --rmi local

echo "Cleanup complete."