# WebChat 테스트 가이드

## 빠른 시작

1. **서버 실행** (터미널 1)
```bash
# 가상환경 활성화
source venv/bin/activate  # Windows: venv\Scripts\activate

# 서버 시작
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **WebChat 테스트** (터미널 2)
```bash
# 테스트 스크립트 실행
python test_webchat.py
```

또는 브라우저에서 직접 접속:
```
http://localhost:8000/static/webchat-test.html
```

## 테스트 가능한 기능

### 1. 기본 대화
- "안녕하세요" - 인사
- "프로젝트 관리를 도와주세요" - 도움 요청
- 일반적인 질문들

### 2. 특수 명령어
- `/help` - 사용 가능한 명령어 목록
- `/clear` - 대화 기록 초기화
- `/buttons` - 버튼 UI 예제
- `/card` - 카드 UI 예제
- `/features` - MOJI 기능 소개

### 3. UI 요소
- **텍스트 메시지**: 일반 대화
- **버튼**: 클릭 가능한 선택지
- **카드**: 이미지와 버튼이 포함된 리치 콘텐츠
- **시스템 메시지**: 연결 상태 등

## 개발자 도구

### WebSocket 연결 확인
브라우저 개발자 도구(F12) > Network 탭에서 WebSocket 연결 확인:
- 연결 상태
- 메시지 송수신 내역

### 콘솔 로그
Console 탭에서 다음 정보 확인:
- WebSocket 연결 상태
- 메시지 송수신 로그
- 에러 메시지

## 문제 해결

### "서버가 실행 중이 아닙니다" 오류
1. FastAPI 서버가 실행 중인지 확인
2. 포트 8000이 사용 가능한지 확인
3. 방화벽 설정 확인

### WebSocket 연결 실패
1. 서버 로그 확인
2. 브라우저 콘솔 에러 확인
3. CORS 설정 확인

### 메시지가 전송되지 않음
1. WebSocket 연결 상태 확인
2. 서버 에러 로그 확인
3. 네트워크 탭에서 WebSocket 메시지 확인

## 고급 테스트

### 부하 테스트
여러 브라우저 탭에서 동시에 접속하여 다중 연결 테스트

### 재연결 테스트
1. 서버 재시작
2. 클라이언트 자동 재연결 확인

### 긴 대화 테스트
연속적인 대화로 컨텍스트 유지 확인

## 커스터마이징

### UI 스타일 변경
`app/static/webchat-test.html`에서 CSS 수정

### 메시지 처리 로직 변경
`app/agents/conversation.py`에서 대화 로직 수정

### WebSocket 설정 변경
`app/static/moji-webchat.js`에서 연결 설정 수정