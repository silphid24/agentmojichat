#!/usr/bin/env python3
"""
WebSocket을 통한 웹챗 기능 테스트 스크립트
"""

import asyncio
import websockets
import json
import sys

async def test_webchat():
    """웹챗 WebSocket 연결 테스트"""
    uri = "ws://localhost:8000/api/v1/adapters/webchat/ws"
    
    try:
        print("🔗 WebSocket 서버에 연결 중...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket 연결 성공!")
            
            # 사용자 정보 설정
            user_info = {
                "type": "user_info",
                "user_id": "test_user_123",
                "user_name": "테스트 사용자"
            }
            
            print(f"📤 사용자 정보 전송: {user_info}")
            await websocket.send(json.dumps(user_info))
            
            # 서버 응답 대기
            response = await websocket.recv()
            print(f"📥 서버 응답: {response}")
            
            # 테스트 메시지 전송
            test_messages = [
                "안녕하세요! 웹챗 테스트입니다.",
                "오늘 날씨는 어떤가요?",
                "AI 에이전트가 잘 작동하나요?"
            ]
            
            for i, message in enumerate(test_messages, 1):
                print(f"\n--- 테스트 메시지 {i} ---")
                
                chat_message = {
                    "type": "message",
                    "content": message,
                    "timestamp": "2025-06-26T13:30:00Z"
                }
                
                print(f"📤 메시지 전송: {message}")
                await websocket.send(json.dumps(chat_message))
                
                # AI 응답 대기
                print("⏳ AI 응답 대기 중...")
                response = await websocket.recv()
                response_data = json.loads(response)
                
                print(f"🤖 AI 응답: {response_data.get('content', response)}")
                
                # 잠시 대기
                await asyncio.sleep(1)
            
            print("\n🎉 모든 테스트 완료!")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ WebSocket 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False
    
    return True

async def main():
    """메인 함수"""
    print("🚀 MOJI WebChat 테스트 시작")
    print("=" * 50)
    
    success = await test_webchat()
    
    print("=" * 50)
    if success:
        print("✅ 테스트 성공!")
        sys.exit(0)
    else:
        print("❌ 테스트 실패!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 