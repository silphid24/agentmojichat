#!/usr/bin/env python3
"""
OpenAI API 키 테스트 스크립트
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_api_key():
    """Test OpenAI API key configuration"""
    print("🔑 OpenAI API 키 설정 확인")
    print("=" * 50)

    try:
        # Load settings
        from app.core.config import settings

        # Check API key
        api_key = settings.llm_api_key
        if not api_key:
            print("❌ API 키가 설정되지 않았습니다!")
            print("\n해결 방법:")
            print("1. .env 파일을 확인하세요")
            print("2. LLM_API_KEY=sk-... 형식으로 설정하세요")
            return False

        # Check format
        if not api_key.startswith("sk-"):
            print("⚠️  API 키 형식이 올바르지 않을 수 있습니다.")
            print("   OpenAI API 키는 'sk-'로 시작해야 합니다.")

        # Show key info (masked)
        masked_key = api_key[:8] + "..." + api_key[-4:]
        print(f"✅ API 키 발견: {masked_key}")
        print(f"   길이: {len(api_key)} 문자")

        # Set environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        print("\n✅ 환경 변수 OPENAI_API_KEY 설정 완료")

        # Test OpenAI connection
        print("\n🔄 OpenAI 연결 테스트...")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            # Simple test
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'API connection successful' in 5 words or less",
                    }
                ],
                max_tokens=10,
            )

            print(f"✅ OpenAI 응답: {response.choices[0].message.content}")
            print("\n🎉 API 키가 올바르게 설정되었습니다!")
            return True

        except Exception as e:
            print(f"❌ OpenAI 연결 실패: {str(e)}")
            print("\n가능한 원인:")
            print("1. API 키가 유효하지 않음")
            print("2. API 크레딧 부족")
            print("3. 네트워크 연결 문제")
            return False

    except Exception as e:
        print(f"❌ 설정 로드 실패: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_api_key()

    if success:
        print("\n\n💡 다음 단계:")
        print("1. 문서 업로드: python3 upload_docs.py")
        print("2. 서버 실행: uvicorn app.main:app --reload")
        print("3. 웹챗 테스트: http://localhost:8000/static/webchat-test.html")
    else:
        print("\n\n⚠️  API 키 문제를 먼저 해결하세요!")
