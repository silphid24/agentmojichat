# Local Development Setup (2024 최신)

## Prerequisites

### Required Software
- Python 3.11 or higher
- PostgreSQL 15
- Redis 7
- Git

### Python Environment Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### faiss-cpu 설치 오류 시

```bash
sudo apt update
sudo apt install swig build-essential python3-dev
pip install --no-cache-dir --only-binary=all faiss-cpu
# 또는 conda 사용
conda install -c conda-forge faiss-cpu
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
DATABASE_URL=postgresql://agentmoji:agentmoji123@localhost:5432/agentmoji
POSTGRES_DB=agentmoji
POSTGRES_USER=agentmoji
POSTGRES_PASSWORD=agentmoji123
REDIS_URL=redis://localhost:6379
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
LLM_API_KEY=your_api_key_here
APP_ENV=development
DEBUG=true
RAG_ENABLED=true
VECTOR_STORE_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Running the Application

1. Apply database migrations:
```bash
alembic upgrade head
```

2. Start the FastAPI server:
```bash
./run_server.sh
# 또는
uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

## WebChat v2 실행

- http://localhost:8100/static/moji-webchat-v2.html

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