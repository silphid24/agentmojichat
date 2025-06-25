# Makefile for MOJI AI Agent

.PHONY: help build start stop clean logs test lint health

help:
	@echo "Available commands:"
	@echo "  make build   - Build Docker images"
	@echo "  make start   - Start development environment"
	@echo "  make stop    - Stop development environment"
	@echo "  make clean   - Clean up everything"
	@echo "  make logs    - View application logs"
	@echo "  make test    - Run tests"
	@echo "  make lint    - Run linters"

build:
	docker-compose -f docker-compose.dev.yml build

start:
	./docker/scripts/start.sh

stop:
	./docker/scripts/stop.sh

clean:
	./docker/scripts/clean.sh

logs:
	docker-compose -f docker-compose.dev.yml logs -f app

test:
	docker-compose -f docker-compose.dev.yml exec app pytest

lint:
	docker-compose -f docker-compose.dev.yml exec app ruff check app/
	docker-compose -f docker-compose.dev.yml exec app black --check app/

shell:
	docker-compose -f docker-compose.dev.yml exec app /bin/bash

db-shell:
	docker-compose -f docker-compose.dev.yml exec postgres psql -U moji -d moji_db

health:
	./docker/scripts/health-check.sh