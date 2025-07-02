#!/bin/bash

echo "📦 MOJI 필수 패키지 설치 스크립트"
echo "=================================="
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3가 설치되지 않았습니다."
    echo "다음 명령으로 pip를 설치하세요:"
    echo "  sudo apt update && sudo apt install python3-pip"
    exit 1
fi

echo "🔧 필수 패키지 설치 중..."
echo ""

# Core packages
pip3 install fastapi uvicorn[standard] python-multipart

# LangChain and AI packages
pip3 install langchain langchain-community langchain-openai openai

# Database and vector store
pip3 install chromadb sentence-transformers

# Additional utilities
pip3 install pydantic pydantic-settings python-jose[cryptography] passlib[bcrypt]
pip3 install aiofiles websockets python-docx

# Development tools
pip3 install pytest pytest-asyncio httpx

echo ""
echo "✅ 패키지 설치 완료!"
echo ""
echo "다음 단계:"
echo "1. API 키 테스트: python3 tools/test_openai_key.py"
echo "2. 문서 업로드: python3 upload_docs.py"
echo "3. 서버 실행: uvicorn app.main:app --reload"