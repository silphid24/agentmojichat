# MOJI AI Agent Testing Guide

이 문서는 MOJI AI Agent 챗봇을 테스트하는 방법을 설명합니다.

## 빠른 시작

### 1. Docker Compose로 실행 (권장)

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 LLM API 키 입력

# 2. Docker 컨테이너 실행
docker-compose up -d

# 3. API 테스트 스크립트 실행
./test_api.sh

# 4. 대화형 테스트
python test_chat.py
```

### 2. 로컬 개발 환경에서 실행

```bash
# 1. Python 가상환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
export LLM_PROVIDER=deepseek
export LLM_MODEL=deepseek-r1
export LLM_API_KEY=your-api-key-here
export DATABASE_URL=postgresql://moji:moji123@localhost:5432/moji_db
export REDIS_URL=redis://localhost:6379
export SECRET_KEY=your-secret-key-here

# 4. 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 테스트 도구

### 1. test_chat.py - 대화형 챗봇 테스터

인터랙티브한 채팅 환경을 제공합니다:

```bash
python test_chat.py
```

기능:
- 자동 로그인/회원가입
- 실시간 대화
- 테스트 시나리오 실행
- 세션 관리

명령어:
- `exit` - 종료
- `clear` - 새 세션 시작
- `test` - 자동 테스트 시나리오 실행

### 2. test_api.sh - API 엔드포인트 테스터

모든 주요 API 엔드포인트를 자동으로 테스트합니다:

```bash
./test_api.sh
```

테스트 항목:
- 서버 헬스 체크
- 사용자 인증
- 채팅 기능
- LLM 상태
- RAG 문서 업로드
- 벡터 스토어
- 에이전트 목록

### 3. Swagger UI - 웹 기반 API 테스터

브라우저에서 API를 테스트할 수 있습니다:

1. http://localhost:8000/docs 접속
2. 우측 상단 "Authorize" 버튼 클릭
3. 토큰 입력 후 API 테스트

## 주요 테스트 시나리오

### 기본 채팅 테스트

```python
# Python 예제
import httpx
import asyncio

async def test_basic_chat():
    # 1. 로그인
    async with httpx.AsyncClient() as client:
        login_resp = await client.post(
            "http://localhost:8000/api/v1/auth/token",
            data={"username": "testuser", "password": "testpass123"}
        )
        token = login_resp.json()["access_token"]
        
        # 2. 채팅
        headers = {"Authorization": f"Bearer {token}"}
        chat_resp = await client.post(
            "http://localhost:8000/api/v1/chat/completions",
            headers=headers,
            json={
                "messages": [
                    {"role": "user", "content": "안녕하세요!"}
                ]
            }
        )
        
        print(chat_resp.json())

asyncio.run(test_basic_chat())
```

### RAG 시스템 테스트

```bash
# 1. 문서 업로드
curl -X POST http://localhost:8000/api/v1/rag/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf"

# 2. RAG 쿼리
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "문서의 주요 내용은?"}'
```

### 벡터 스토어 테스트

```bash
# 1. 벡터 스토어 생성
curl -X POST http://localhost:8000/api/v1/vectorstore/stores \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "test_store",
    "store_type": "chroma",
    "collection_name": "test_collection"
  }'

# 2. 문서 추가
curl -X POST http://localhost:8000/api/v1/vectorstore/documents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["테스트 문서입니다."],
    "store_id": "test_store"
  }'

# 3. 검색
curl -X POST http://localhost:8000/api/v1/vectorstore/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "테스트",
    "store_id": "test_store"
  }'
```

## 문제 해결

### 서버가 실행되지 않을 때

1. Docker가 실행 중인지 확인
```bash
docker ps
```

2. 로그 확인
```bash
docker-compose logs -f app
```

3. 포트 충돌 확인
```bash
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows
```

### LLM 연결 오류

1. API 키 확인
```bash
echo $LLM_API_KEY
```

2. LLM 상태 확인
```bash
curl http://localhost:8000/api/v1/llm/status -H "Authorization: Bearer $TOKEN"
```

3. 다른 LLM provider로 변경
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-3.5-turbo
export LLM_API_KEY=your-openai-key
```

### 데이터베이스 연결 오류

1. PostgreSQL 컨테이너 확인
```bash
docker-compose ps postgres
```

2. 데이터베이스 초기화
```bash
docker-compose down -v
docker-compose up -d
```

## 성능 테스트

### 부하 테스트 (locust 사용)

```python
# locustfile.py
from locust import HttpUser, task, between

class ChatUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # 로그인
        response = self.client.post("/api/v1/auth/token", 
            data={"username": "testuser", "password": "testpass123"})
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task
    def chat(self):
        self.client.post("/api/v1/chat/completions",
            headers=self.headers,
            json={
                "messages": [{"role": "user", "content": "Hello!"}]
            })

# 실행: locust -f locustfile.py --host=http://localhost:8000
```

## 통합 테스트

전체 시스템 통합 테스트:

```bash
# 1. 전체 테스트 스위트 실행
pytest tests/ -v

# 2. 특정 테스트만 실행
pytest tests/test_chat.py -v
pytest tests/test_vectorstore.py -v

# 3. 커버리지 확인
pytest --cov=app tests/
```

## 모니터링

### 로그 확인

```bash
# 전체 로그
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f app
docker-compose logs -f postgres
```

### 메트릭 확인

API 응답 시간과 상태는 각 요청의 응답 헤더에서 확인할 수 있습니다:

```bash
curl -w "\n응답시간: %{time_total}s\n" \
  http://localhost:8000/api/v1/health
```

## 추가 리소스

- [API 문서](http://localhost:8000/docs)
- [ReDoc API 문서](http://localhost:8000/redoc)
- [OpenAPI Schema](http://localhost:8000/openapi.json)