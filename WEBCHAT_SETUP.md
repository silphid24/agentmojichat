# MOJI WebChat 설정 가이드

## 필수 준비사항

### 1. Python 환경 설정
```bash
# Python 3.11+ 및 pip 설치 확인
python3 --version
pip3 --version

# 가상환경 생성 (권장)
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 2. 의존성 패키지 설치
```bash
# 전체 패키지 설치
pip install -r requirements.txt

# 또는 핵심 패키지만 설치
pip install fastapi uvicorn langchain langchain-openai openai websockets
```

### 3. OpenAI API 키 설정
`.env` 파일에서 실제 API 키로 변경:
```
LLM_API_KEY=sk-your-actual-openai-api-key-here
```

## 서버 실행 방법

### 방법 1: 실행 스크립트 사용
```bash
./run_server.sh
```

### 방법 2: 직접 실행
```bash
# 가상환경 활성화 후
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 방법 3: Python 모듈로 실행
```bash
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 웹챗 접속

서버가 실행되면 브라우저에서 다음 주소로 접속:
```
http://localhost:8000/static/webchat-test.html
```

## 테스트 가능한 기능

1. **기본 대화**
   - "안녕하세요"
   - "무엇을 도와드릴까요?"
   - 일반적인 질문들

2. **특수 명령어**
   - `/help` - 도움말
   - `/clear` - 대화 초기화
   - `/buttons` - 버튼 예제
   - `/card` - 카드 UI 예제

3. **파일 첨부** (구현됨)
   - 이미지, 문서 등 업로드 가능

## 문제 해결

### "ModuleNotFoundError" 오류
```bash
# 필요한 패키지 설치
pip install [missing-module-name]
```

### "Connection refused" 오류
- 서버가 실행 중인지 확인
- 포트 8000이 사용 가능한지 확인
- 방화벽 설정 확인

### API 키 오류
- `.env` 파일의 `LLM_API_KEY`가 올바른지 확인
- OpenAI 계정의 API 키 유효성 확인

## 개발 환경 요구사항

- Python 3.11 이상
- pip 패키지 관리자
- 가상환경 (venv) 사용 권장
- OpenAI API 키 (ChatGPT 사용을 위해)

## 다음 단계

1. 서버가 정상적으로 실행되면 웹챗 인터페이스에서 MOJI와 대화 가능
2. 개발자 도구(F12)를 열어 WebSocket 연결 상태 확인
3. 다양한 명령어와 대화를 테스트