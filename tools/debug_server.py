#!/usr/bin/env python3
"""
MOJI 서버 디버그 테스트 스크립트
문제를 진단하고 해결하기 위한 테스트
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_llm_router():
    """LLM Router 직접 테스트"""
    print("🔍 LLM Router 테스트 시작...")
    
    try:
        from app.llm.router import llm_router
        from langchain_core.messages import HumanMessage
        
        # 초기화
        print("- LLM Router 초기화 중...")
        await llm_router.initialize()
        print("✅ 초기화 완료")
        
        # 현재 설정 확인
        print(f"- 현재 Provider: {llm_router.config.provider}")
        print(f"- 현재 Model: {llm_router.config.model}")
        print(f"- API Key 설정: {'✅' if llm_router.config.api_key else '❌'}")
        
        # 테스트 메시지 생성
        print("\n📤 테스트 메시지 전송...")
        messages = [HumanMessage(content="안녕하세요! 테스트입니다.")]
        
        # 응답 생성
        response = await llm_router.generate(messages=messages)
        print(f"📥 응답: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_env_settings():
    """환경 설정 확인"""
    print("\n🔍 환경 설정 확인...")
    
    try:
        from app.core.config import settings
        
        print(f"- LLM Provider: {settings.llm_provider}")
        print(f"- LLM Model: {settings.llm_model}")
        print(f"- API Key 길이: {len(settings.llm_api_key) if settings.llm_api_key else 0}")
        print(f"- API Base: {settings.llm_api_base}")
        
        # .env 파일 확인
        env_path = Path(".env")
        if env_path.exists():
            print(f"✅ .env 파일 존재: {env_path.absolute()}")
            
            # API 키 확인
            with open(env_path, 'r') as f:
                content = f.read()
                if 'sk-' in content:
                    print("✅ OpenAI API 키 형식 확인됨")
                else:
                    print("⚠️ OpenAI API 키가 올바른 형식이 아닙니다")
        else:
            print("❌ .env 파일이 없습니다")
            
    except Exception as e:
        print(f"❌ 설정 확인 실패: {str(e)}")

async def main():
    """메인 테스트 함수"""
    print("🤖 MOJI 서버 디버그 테스트")
    print("=" * 50)
    
    # 환경 설정 확인
    await test_env_settings()
    
    # LLM Router 테스트
    success = await test_llm_router()
    
    if success:
        print("\n✅ 모든 테스트 통과! 서버를 다시 시작해보세요.")
        print("\n다음 명령으로 서버 실행:")
        print("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n❌ 테스트 실패. 위의 오류 메시지를 확인하세요.")
        print("\n가능한 해결 방법:")
        print("1. .env 파일의 LLM_API_KEY가 올바른지 확인")
        print("2. OpenAI API 키가 유효한지 확인")
        print("3. 인터넷 연결 상태 확인")

if __name__ == "__main__":
    asyncio.run(main())