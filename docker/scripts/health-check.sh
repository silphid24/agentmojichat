#!/bin/bash

echo "Checking MOJI AI Agent services health..."

# Check if services are running
echo -n "Checking Docker containers... "
if docker-compose -f docker-compose.dev.yml ps | grep -q "Up"; then
    echo "✓"
else
    echo "✗"
    echo "Some services are not running. Run 'make start' to start them."
    exit 1
fi

# Check app health
echo -n "Checking FastAPI app... "
if curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "✓"
    echo "  API Response: $(curl -s http://localhost:8000/api/v1/health | python3 -m json.tool)"
else
    echo "✗"
    echo "  FastAPI is not responding"
fi

# Check PostgreSQL
echo -n "Checking PostgreSQL... "
if docker-compose -f docker-compose.dev.yml exec -T postgres pg_isready -U moji > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗"
    echo "  PostgreSQL is not ready"
fi

# Check Redis
echo -n "Checking Redis... "
if docker-compose -f docker-compose.dev.yml exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "✓"
else
    echo "✗"
    echo "  Redis is not responding"
fi

echo ""
echo "Health check complete!"