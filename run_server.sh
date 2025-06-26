#!/bin/bash

# MOJI WebChat 서버 실행 스크립트
# 
# 사용법:
# 1. 이 스크립트에 실행 권한 부여: chmod +x run_server.sh
# 2. 실행: ./run_server.sh

echo "MOJI WebChat 서버를 시작합니다..."
echo ""
echo "============================================="
echo "필수 사항:"
echo "1. Python 3.11+ 설치"
echo "2. pip 설치" 
echo "3. 가상환경 설정 (권장)"
echo "============================================="
echo ""

# 가상환경이 있는지 확인
if [ -d "venv" ]; then
    echo "가상환경을 활성화합니다..."
    source venv/bin/activate
else
    echo "가상환경이 없습니다. 다음 명령으로 생성하세요:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
fi

# 필요한 패키지 설치 확인
echo "필요한 패키지를 확인합니다..."
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo ""
    echo "필요한 패키지가 설치되지 않았습니다."
    echo "다음 명령으로 설치하세요:"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "또는 주요 패키지만 설치:"
    echo "  pip install fastapi uvicorn langchain openai"
    exit 1
fi

# OpenAI API 키 확인
if [ -z "$LLM_API_KEY" ]; then
    if grep -q "your-openai-api-key-here" .env 2>/dev/null; then
        echo ""
        echo "⚠️  경고: OpenAI API 키가 설정되지 않았습니다!"
        echo ".env 파일에서 LLM_API_KEY를 실제 API 키로 변경하세요."
        echo ""
    fi
fi

# 서버 시작
echo "FastAPI 서버를 시작합니다..."
echo "접속 URL: http://localhost:8000/static/webchat-test.html"
echo ""
echo "서버를 중지하려면 Ctrl+C를 누르세요."
echo "============================================="
echo ""

# uvicorn 실행
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000