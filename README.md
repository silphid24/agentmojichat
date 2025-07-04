# MOJI AI Agent

## 🚀 프로젝트 개요

MOJI는 다양한 플랫폼에서 활용 가능한 지능형 AI 어시스턴트입니다.

### 주요 특징

- **다중 플랫폼 지원**: Slack, Teams, KakaoTalk, Discord, Web Chat
- **지능형 대화 엔진**: LangChain 기반 대화 처리
- **고급 RAG 시스템**: 하이브리드 검색, 리랭킹, 적응형 기능
- **확장 가능한 LLM**: DeepSeek, OpenAI, Anthropic 등 다양한 모델 지원
- **성능 최적화**: 모델 예열, 캐싱, 병렬 처리
- **플러그인 시스템**: 동적 기능 확장

## 📋 프로젝트 구조

```
agentmoji/
├── app/                # 애플리케이션 코드
│   ├── adapters/      # 플랫폼 어댑터
│   ├── agents/        # AI 에이전트
│   ├── api/           # REST API
│   ├── core/          # 핵심 모듈
│   ├── llm/           # LLM 프로바이더
│   ├── rag/           # RAG 시스템
│   └── vectorstore/   # 벡터 스토어
├── data/              # 데이터 디렉토리
├── tests/             # 테스트 코드
├── script/            # 개발 작업 설명서
└── docs/              # 문서
```

## 🛠 기술 스택

- Python 3.11+
- FastAPI 0.111
- LangChain 0.2.x, LangGraph 0.1.x
- PostgreSQL 15, Redis 7, Chroma DB
- DeepSeek, OpenAI, Anthropic, 커스텀 LLM 서버
- 하이브리드 검색, 리랭킹, 적응형 기능, 캐싱, 병렬 처리

## 🚀 시작하기

### 1. 환경 설정

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### faiss-cpu 설치 오류 시

```bash
sudo apt update
sudo apt install swig build-essential python3-dev
pip install --no-cache-dir --only-binary=all faiss-cpu
# 또는 conda 사용
conda install -c conda-forge faiss-cpu
```

### 2. 환경 변수 설정

.env 파일 예시:
```env
DEBUG=true
PORT=8100
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
LLM_API_KEY=your-api-key
DATABASE_URL=postgresql://user:pass@localhost/moji
REDIS_URL=redis://localhost:6379
RAG_ENABLED=true
VECTOR_STORE_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CACHE_ENABLED=true
MODEL_WARMUP=true
PARALLEL_SEARCH=true
```

### 3. 서버 실행

```bash
./run_server.sh
# 또는
uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

서버가 실행되면:
- API: http://localhost:8100
- 문서: http://localhost:8100/docs
- WebChat(v2): http://localhost:8100/static/moji-webchat-v2.html

## 📡 API 사용법

### 하이브리드 검색 예시

```bash
curl -X POST http://localhost:8100/api/v1/rag/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "검색할 내용",
    "top_k": 5,
    "use_reranking": true
  }'
```

## 🔌 플랫폼 통합, 🤖 LLM 프로바이더 설정, 🧪 테스트, 📚 문서, 성능 최적화, 기여 방법 등은 기존 내용과 동일하게 유지하되, 최신 기능 위주로 보강

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 🙏 감사의 말

- LangChain 커뮤니티
- FastAPI 개발팀
- 모든 오픈소스 기여자들