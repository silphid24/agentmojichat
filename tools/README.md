# Developer Tools

This directory contains utility scripts and tools for development and testing of the MOJI AI Agent.

## Utility Scripts

### Interactive Testing Tools

- **interactive_chat.py** (formerly test_chat.py)
  - Interactive CLI tool for testing the MOJI chatbot
  - Allows real-time conversation with the agent
  - Useful for manual testing and debugging

- **debug_server.py** (formerly test_server_debug.py)
  - FastAPI server debug utility
  - Provides detailed request/response logging
  - Helps debug API endpoint issues

- **quick_test.py**
  - Quick web-based test launcher
  - Automatically opens browser with webchat interface
  - Checks requirements and API keys before starting

### WebSocket Testing

- **test_websocket.py**
  - WebSocket connection tester
  - Tests real-time communication features
  - Useful for debugging WebSocket-based chat

- **test_webchat.py**
  - Web chat interface tester
  - Tests the browser-based chat UI
  - Validates frontend-backend integration

### API Testing

- **test_api.sh**
  - Bash script for testing API endpoints
  - Includes examples for auth, chat, and document operations
  - Quick way to verify API functionality

- **test_openai_key.py**
  - API key validation tool
  - Tests LLM provider connectivity
  - Verifies environment configuration

## Usage

All Python scripts can be run directly:
```bash
python tools/interactive_chat.py
python tools/debug_server.py
python tools/test_openai_key.py
```

For the bash script:
```bash
chmod +x tools/test_api.sh
./tools/test_api.sh
```

## Note

These are development utilities and should not be used in production.
For unit tests, see the `tests/` directory.