# MOJI 웹챗 완전 가이드

MOJI 웹챗 인터페이스 설정 및 테스트를 위한 종합 가이드입니다.

## 사전 요구사항

- Python 3.11 이상
- pip 패키지 관리자
- 2GB 이상의 가용 RAM
- 최신 웹 브라우저 (Chrome, Firefox, Safari, Edge)

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정

프로젝트 루트에 `.env` 파일 생성:

```bash
# LLM 설정
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
LLM_API_KEY=your-api-key-here
LLM_API_BASE=https://api.deepseek.com

# 서버 설정
APP_ENV=development
DEBUG=true
```

### 3. 서버 시작

#### 옵션 1: 빠른 테스트 (권장)
```bash
python tools/quick_test.py
```
이 명령은:
- 모든 요구사항 확인
- API 키 검증
- 서버 시작
- 브라우저에서 웹챗 자동 열기

#### 옵션 2: 수동 시작
```bash
# uvicorn 직접 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는 실행 스크립트 사용
./run_server.sh
```

### 4. 웹챗 접속

브라우저에서 다음 주소로 이동:
- 웹챗: http://localhost:8000/static/webchat-test.html
- API 문서: http://localhost:8000/docs
- 상태 확인: http://localhost:8000/health

## 웹챗 기능

### 기본 대화
- 입력 필드에 메시지 입력
- Enter 키 또는 전송 버튼 클릭
- 타이핑 애니메이션과 함께 AI 응답 표시
- 메시지는 마크다운 포맷 지원

### 특수 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `/help` | 사용 가능한 명령어 표시 | `/help` |
| `/clear` | 대화 기록 삭제 | `/clear` |
| `/buttons` | 대화형 버튼 표시 | `/buttons` |
| `/card` | 카드 UI 예제 표시 | `/card` |
| `/rag` | 지식 베이스 검색 | `/rag MOJI가 무엇인가요?` |
| `/rag-help` | RAG 시스템 도움말 | `/rag-help` |
| `/rag-stats` | RAG 통계 표시 | `/rag-stats` |

### UI 요소

- **챗 헤더**: "MOJI WebChat Test"와 연결 상태 표시
- **메시지 영역**: user/assistant 레이블과 함께 대화 표시
- **입력 영역**: 텍스트 입력 필드와 전송 버튼
- **상태 표시기**: 타이핑, 로딩, 오류 상태 표시

## 개발자 테스트

### 브라우저 콘솔
F12를 눌러 개발자 도구를 열고 모니터링:
- WebSocket 연결
- API 요청/응답
- JavaScript 오류
- 성능 지표

### API 테스트
```bash
# 채팅 엔드포인트 테스트
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕하세요 MOJI"}'

# 상태 확인
curl http://localhost:8000/health
```

### WebSocket 테스트
```bash
python tools/test_websocket.py
```

## 문제 해결

### 일반적인 문제

1. **연결 거부**
   - 서버 실행 확인: `ps aux | grep uvicorn`
   - 포트 사용 가능 확인: `lsof -i :8000`
   - 방화벽 설정 확인

2. **모듈을 찾을 수 없음**
   - 누락된 패키지 설치: `pip install -r requirements.txt`
   - 가상환경 활성화: `source venv/bin/activate`

3. **API 키 오류**
   - `.env` 파일 존재 확인
   - API 키 유효성 확인
   - 테스트: `python tools/test_openai_key.py`

4. **WebSocket 연결 실패**
   - 브라우저 콘솔에서 오류 확인
   - 브라우저의 WebSocket 지원 확인
   - 다른 브라우저나 시크릿 모드 시도

### 디버그 모드

상세 로깅 활성화:
```bash
# .env에 설정
DEBUG=true
LOG_LEVEL=debug

# 또는 디버그 서버 실행
python tools/debug_server.py
```

## 성능 최적화

### 서버 튜닝
```bash
# 워커를 사용한 프로덕션 서버
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### 클라이언트 최적화
- 브라우저 캐싱 활성화
- API 호출 최소화
- 연결 풀링 사용

## 고급 사용법

### 커스텀 LLM 프로바이더
```bash
# .env 파일에서
LLM_PROVIDER=custom
LLM_API_BASE=https://your-llm-endpoint.com
LLM_MODEL=your-model-name
```

### 다중 대화
- 여러 브라우저 탭 열기
- 각 탭은 별도의 대화 유지
- 동시 연결 테스트

### 통합 테스트
```bash
# 모든 테스트 실행
pytest tests/

# 특정 웹챗 테스트 실행
pytest tests/test_chat.py -v
```

## 지원

- 로그 확인: `tail -f logs/app.log`
- 이슈 보고: https://github.com/your-repo/issues
- 문서: `/docs` 폴더

---

RAG 관련 테스트는 [RAG_GUIDE.md](RAG_GUIDE.md)를 참조하세요.