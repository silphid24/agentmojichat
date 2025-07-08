#!/usr/bin/env python3
"""
WebChat 테스트 스크립트
웹 브라우저에서 챗봇을 테스트할 수 있습니다.
"""

import webbrowser
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """WebChat 테스트 실행"""
    print("🤖 MOJI WebChat 테스트를 시작합니다...")
    print("-" * 50)

    # Check if server is running
    import requests

    try:
        response = requests.get("http://localhost:8000/api/v1/health")
        if response.status_code != 200:
            print("❌ FastAPI 서버가 실행 중이 아닙니다!")
            print("다음 명령으로 서버를 먼저 실행하세요:")
            print("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
            return
    except requests.ConnectionError:
        print("❌ FastAPI 서버가 실행 중이 아닙니다!")
        print("다음 명령으로 서버를 먼저 실행하세요:")
        print("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return

    print("✅ FastAPI 서버가 실행 중입니다.")
    print()

    # WebChat test URL
    url = "http://localhost:8000/static/webchat-test.html"

    print("📱 WebChat 테스트 페이지를 열고 있습니다...")
    print(f"URL: {url}")
    print()

    # Open browser
    webbrowser.open(url)

    print("🎯 테스트 방법:")
    print("1. 웹 브라우저에서 채팅창이 열립니다")
    print("2. 메시지를 입력하고 전송 버튼을 클릭하거나 Enter를 누르세요")
    print("3. 다음 명령어들을 시도해보세요:")
    print("   - /help - 도움말 보기")
    print("   - /buttons - 버튼 예제")
    print("   - /card - 카드 예제")
    print("   - /features - MOJI 기능 소개")
    print("   - 안녕하세요 - 일반 대화")
    print()
    print("💡 팁:")
    print("- WebSocket 연결 상태가 상단에 표시됩니다")
    print("- 연결이 끊어지면 자동으로 재연결을 시도합니다")
    print("- 개발자 도구(F12)에서 콘솔 로그를 확인할 수 있습니다")
    print()
    print("종료하려면 Ctrl+C를 누르세요...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 WebChat 테스트를 종료합니다.")


if __name__ == "__main__":
    main()
