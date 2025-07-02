# MOJI AI Agent

## 🚀 프로젝트 개요

MOJI는 다양한 플랫폼에서 활용 가능한 지능형 AI 어시스턴트입니다. 

### 주요 특징

- **다중 플랫폼 지원**: Slack, Microsoft Teams, KakaoTalk, Discord, Web Chat
- **지능형 대화 엔진**: LangChain 기반 대화 처리
- **RAG 시스템**: 문서 기반 지식 검색 및 활용
- **확장 가능한 LLM**: DeepSeek, OpenAI, Anthropic 등 다양한 모델 지원
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

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI 0.111
- **AI/ML**: LangChain 0.2.x, LangGraph 0.1.x
- **Database**: PostgreSQL 15, Redis 7, Chroma DB
- **LLM**: DeepSeek R1 (기본), OpenAI, Anthropic, 커스텀 LLM 서버 지원

## 🚀 시작하기

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```env
# Application
DEBUG=true
PORT=8100

# LLM Configuration
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
LLM_API_KEY=your-api-key

# 워크스테이션 LLM 서버 사용 예시
# LLM_PROVIDER=custom
# LLM_MODEL=your-model-name
# LLM_API_BASE=http://192.168.0.7:5000/v1
# LLM_API_KEY=your-api-key-if-needed

# Database
DATABASE_URL=postgresql://user:pass@localhost/moji
REDIS_URL=redis://localhost:6379
```

### 3. 서버 실행

```bash
./run_server.sh
```

서버가 실행되면:
- API: http://localhost:8100
- 문서: http://localhost:8100/docs
- WebChat: http://localhost:8100/static/webchat-test.html

## 📡 API 사용법

### 채팅 엔드포인트

```bash
curl -X POST http://localhost:8100/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "message": "안녕하세요!",
    "session_id": "test-session"
  }'
```

### RAG 문서 추가

```bash
curl -X POST http://localhost:8100/api/v1/rag/add/text \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "text": "문서 내용...",
    "metadata": {"source": "manual"}
  }'
```

## 🔌 플랫폼 통합

### Slack 연동

1. Slack App 생성
2. 환경 변수 설정:
   ```env
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_APP_TOKEN=xapp-...
   ```
3. 어댑터 활성화

### Web Chat 위젯

```html
<script src="http://localhost:8100/static/moji-widget.js"></script>
<script>
  MojiChat.init({
    apiUrl: 'http://localhost:8100',
    position: 'bottom-right'
  });
</script>
```

## 🤖 LLM 프로바이더 설정

MOJI는 다양한 LLM 프로바이더를 지원합니다:

### DeepSeek (기본)
```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
LLM_API_KEY=your-deepseek-api-key
```

### OpenAI
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-openai-api-key
```

### 워크스테이션 LLM 서버
로컬 워크스테이션에서 실행 중인 LLM 서버를 사용할 수 있습니다:

```env
LLM_PROVIDER=custom
LLM_MODEL=your-model-name
LLM_API_BASE=http://192.168.0.7:5000/v1
LLM_API_KEY=your-api-key-if-needed  # 선택사항
```

### Ollama 등 로컬 모델
```env
LLM_PROVIDER=custom
LLM_MODEL=llama3
LLM_API_BASE=http://localhost:11434/v1
```

## 🧪 테스트

```bash
# 단위 테스트
pytest tests/unit

# 통합 테스트
pytest tests/integration

# 전체 테스트
pytest
```

## 📚 문서

- [API 문서](http://localhost:8100/docs)
- [개발 가이드](./CLAUDE.md)
- [로컬 환경 설정](./LOCAL_SETUP.md)
- [WebChat 실행 가이드](./WEBCHAT_GUIDE.md)
- [RAG 테스트 가이드](./RAG_GUIDE.md)


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