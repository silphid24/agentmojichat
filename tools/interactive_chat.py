#!/usr/bin/env python3
"""
Interactive chatbot testing script for MOJI AI Agent
"""

import asyncio
import httpx
import os
from typing import Optional


class ChatbotTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.session_id: Optional[str] = None
        self.headers = {}

    async def register(self, username: str, password: str) -> bool:
        """Register a new user"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json={"username": username, "password": password},
                )
                if response.status_code == 200:
                    print(f"✅ User '{username}' registered successfully")
                    return True
                else:
                    print(f"❌ Registration failed: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ Connection error: {e}")
                return False

    async def login(self, username: str, password: str) -> bool:
        """Login and get access token"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/auth/token",
                    data={"username": username, "password": password},
                )
                if response.status_code == 200:
                    data = response.json()
                    self.token = data["access_token"]
                    self.headers = {"Authorization": f"Bearer {self.token}"}
                    print(f"✅ Logged in as '{username}'")
                    return True
                else:
                    print(f"❌ Login failed: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ Connection error: {e}")
                return False

    async def create_session(self) -> bool:
        """Create a new chat session"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat/sessions",
                    headers=self.headers,
                    json={},
                )
                if response.status_code == 200:
                    data = response.json()
                    self.session_id = data["id"]
                    print(f"✅ Created session: {self.session_id[:8]}...")
                    return True
                else:
                    print(f"❌ Failed to create session: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ Error creating session: {e}")
                return False

    async def send_message(self, message: str) -> Optional[str]:
        """Send a message to the chatbot"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/chat/completions",
                    headers=self.headers,
                    json={
                        "messages": [{"role": "user", "content": message}],
                        "session_id": self.session_id,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f"❌ Error: {response.text}")
                    return None
            except httpx.TimeoutException:
                print("❌ Request timed out. The server might be processing...")
                return None
            except Exception as e:
                print(f"❌ Error sending message: {e}")
                return None

    async def test_endpoints(self):
        """Test various endpoints"""
        print("\n🔍 Testing API endpoints...")

        async with httpx.AsyncClient() as client:
            # Test health endpoint
            try:
                response = await client.get(f"{self.base_url}/api/v1/health")
                if response.status_code == 200:
                    print("✅ Health check: OK")
                else:
                    print("❌ Health check failed")
            except httpx.ConnectError:
                print("❌ Cannot connect to server")
                return False

            # Test LLM status
            if self.token:
                try:
                    response = await client.get(
                        f"{self.base_url}/api/v1/llm/status", headers=self.headers
                    )
                    if response.status_code == 200:
                        data = response.json()
                        print(f"✅ LLM Provider: {data.get('provider', 'unknown')}")
                        print(f"✅ LLM Model: {data.get('model', 'unknown')}")
                except (httpx.HTTPError, KeyError):
                    print("⚠️  LLM status endpoint not available")

        return True

    async def interactive_chat(self):
        """Start interactive chat session"""
        print("\n💬 Starting interactive chat...")
        print("Type 'exit' to quit, 'clear' to clear history, 'test' to run tests")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() == "exit":
                    print("👋 Goodbye!")
                    break

                if user_input.lower() == "clear":
                    await self.create_session()
                    print("🔄 Started new session")
                    continue

                if user_input.lower() == "test":
                    await self.run_test_scenarios()
                    continue

                if not user_input:
                    continue

                print("🤔 MOJI is thinking...")
                response = await self.send_message(user_input)

                if response:
                    print(f"\n🤖 MOJI: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    async def run_test_scenarios(self):
        """Run predefined test scenarios"""
        print("\n🧪 Running test scenarios...")

        test_cases = [
            "안녕하세요! 자기소개를 해주세요.",
            "파이썬에서 리스트 컴프리헨션의 장점은 무엇인가요?",
            "오늘 날씨가 어때?",
            "1부터 10까지의 합을 계산해줘",
            "FastAPI와 Flask의 차이점을 설명해줘",
        ]

        for i, test_message in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_message}")
            response = await self.send_message(test_message)
            if response:
                print(f"✅ Response: {response[:100]}...")
            else:
                print("❌ No response")

            await asyncio.sleep(1)  # Rate limiting


async def main():
    print("🚀 MOJI AI Agent Chatbot Tester")
    print("=" * 50)

    # Check environment variables
    base_url = os.getenv("MOJI_BASE_URL", "http://localhost:8000")

    tester = ChatbotTester(base_url)

    # Test endpoints first
    if not await tester.test_endpoints():
        print("\n❌ Server is not running. Please start the server first:")
        print("   docker-compose up -d")
        print("   or")
        print("   uvicorn app.main:app --reload")
        return

    # Login flow
    print("\n🔐 Authentication")
    username = input("Username (or press Enter for 'testuser'): ").strip() or "testuser"
    password = (
        input("Password (or press Enter for 'testpass123'): ").strip() or "testpass123"
    )

    # Try to login first
    if not await tester.login(username, password):
        # If login fails, try to register
        print("\n📝 Attempting to register new user...")
        if await tester.register(username, password):
            if not await tester.login(username, password):
                print("❌ Failed to login after registration")
                return
        else:
            print("❌ Failed to register and login")
            return

    # Create session
    if not await tester.create_session():
        print("❌ Failed to create chat session")
        return

    # Start interactive chat
    await tester.interactive_chat()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
