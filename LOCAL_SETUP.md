# Local Development Setup (Without Docker)

## Prerequisites

### Required Software
- Python 3.11 or higher
- PostgreSQL 15
- Redis 7
- Git

### Python Environment Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Database Setup

### PostgreSQL
1. Install PostgreSQL 15 locally
2. Create database and user:
```sql
CREATE DATABASE agentmoji;
CREATE USER agentmoji WITH PASSWORD 'agentmoji123';
GRANT ALL PRIVILEGES ON DATABASE agentmoji TO agentmoji;
```

### Redis
1. Install Redis 7 locally
2. Start Redis server:
```bash
redis-server
```

## Environment Variables

Create a `.env` file in the project root:
```env
# Database
DATABASE_URL=postgresql://agentmoji:agentmoji123@localhost:5432/agentmoji
POSTGRES_DB=agentmoji
POSTGRES_USER=agentmoji
POSTGRES_PASSWORD=agentmoji123

# Redis
REDIS_URL=redis://localhost:6379

# LLM Configuration
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
DEEPSEEK_API_KEY=your_api_key_here

# Application
APP_ENV=development
LOG_LEVEL=INFO
```

## Running the Application

1. Apply database migrations:
```bash
alembic upgrade head
```

2. Start the FastAPI server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing

Run tests without Docker:
```bash
pytest tests/
```

## Development Tips

- Use `pip install -e .` for editable installation
- Set up pre-commit hooks: `pre-commit install`
- For hot reloading, ensure `--reload` flag is used with uvicorn

## Note

Docker setup is available for future use in the `docker-future/` directory when needed.