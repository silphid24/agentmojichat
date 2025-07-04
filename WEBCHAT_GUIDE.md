# MOJI 웹챗 완전 가이드 (최신화)

MOJI 웹챗 인터페이스 설정, 고급 기능, 성능 최적화, 문제 해결까지 한 번에 안내합니다.

---

## 사전 요구사항

- Python 3.11 이상
- pip 패키지 관리자
- 2GB 이상 RAM
- 최신 웹 브라우저 (Chrome, Firefox, Safari, Edge)

---

## 빠른 시작

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

---

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
LLM_API_KEY=your-api-key
APP_ENV=development
DEBUG=true
RAG_ENABLED=true
VECTOR_STORE_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

### 3. 서버 시작

```bash
./run_server.sh
# 또는
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### 4. 웹챗 접속

- 웹챗: http://localhost:8000/static/webchat-test.html
- API 문서: http://localhost:8000/docs
- 상태 확인: http://localhost:8000/health

---

## 웹챗 주요 기능

### 기본 대화

- 입력 필드에 메시지 입력, Enter 또는 전송 버튼 클릭
- AI 응답은 마크다운 지원, 타이핑 애니메이션 제공

### 특수 명령어

| 명령어      | 설명                        | 예시                        |
|-------------|-----------------------------|-----------------------------|
| `/help`     | 사용 가능한 명령어 표시      | `/help`                     |
| `/clear`    | 대화 기록 삭제              | `/clear`                    |
| `/buttons`  | 대화형 버튼 표시            | `/buttons`                  |
| `/card`     | 카드 UI 예제 표시           | `/card`                     |
| `/rag`      | 지식 베이스 검색            | `/rag MOJI가 무엇인가요?`   |
| `/rag-help` | RAG 시스템 도움말           | `/rag-help`                 |
| `/rag-stats`| RAG 통계 표시               | `/rag-stats`                |

### 하이브리드 검색 및 RAG

- `/rag` 명령어로 벡터+키워드 하이브리드 검색 지원
- 신뢰도 점수, 출처 인용, 다중 문서 통합 등 고급 RAG 기능 제공

### WebSocket 실시간 대화

- WebSocket 기반 실시간 대화 지원
- 연결 상태, 오류, 재연결 등 UI에서 실시간 표시

### UI 요소

- 챗 헤더: "MOJI WebChat Test" 및 연결 상태
- 메시지 영역: user/assistant 레이블, 마크다운 지원
- 입력 영역: 텍스트 필드, 전송 버튼
- 상태 표시기: 타이핑, 로딩, 오류

---

## 개발자/테스트 가이드

### 브라우저 콘솔

- F12로 개발자 도구 열고 WebSocket, API 요청/응답, JS 오류, 성능 지표 확인

### API 테스트

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕하세요 MOJI"}'
```

### WebSocket 테스트

```bash
python tools/test_websocket.py
```

---

## 문제 해결

1. **연결 거부**: 서버 실행, 포트, 방화벽 확인
2. **모듈 없음**: `pip install -r requirements.txt`, 가상환경 활성화
3. **API 키 오류**: .env 파일, 키 유효성, `python tools/test_openai_key.py`
4. **WebSocket 실패**: 브라우저 콘솔, 지원 여부, 시크릿 모드 시도

### 디버그 모드

```bash
# .env에 설정
DEBUG=true
LOG_LEVEL=debug
python tools/debug_server.py
```

---

## 성능 최적화

- 서버: uvicorn --workers 4 등 멀티 워커 사용
- 클라이언트: 브라우저 캐싱, API 호출 최소화, 연결 풀링
- 서버/클라이언트 모두 실시간 성능 모니터링 지원

---

## 고급 사용법

- 커스텀 LLM 프로바이더, 다중 대화, 통합 테스트 등 지원
- WebChat v2: 모델/프로바이더 실시간 전환, 고급 UI, 상태 표시 강화

---

## 지원

- 로그: `tail -f logs/app.log`
- 이슈: https://github.com/your-repo/issues
- 문서: `/docs` 폴더, [RAG_GUIDE.md](RAG_GUIDE.md) 참고
RAG 관련 테스트는 [RAG_GUIDE.md](RAG_GUIDE.md)를 참조하세요.