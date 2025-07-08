# Developer Tools

This directory contains utility scripts and tools for development, testing, and document management of the MOJI AI Agent.

## Document Management Tools

- **upload_docs.py**
  - Document upload and indexing tool
  - Supports incremental updates and batch processing
  - Handles multiple file formats (.txt, .md, .docx, .pdf)

- **clear_and_reload_docs.py**
  - Vector store initialization and re-indexing tool
  - Clears existing indices and rebuilds from scratch
  - Includes verification and stats reporting

- **vector_db_manager.py**
  - Vector database management utility
  - Backup/restore functionality
  - Database statistics and cleanup operations

- **manage_docs.py**
  - Comprehensive document management interface
  - Document listing, searching, and metadata management
  - Index backup and restore capabilities

- **rag_health_check.py**
  - RAG system health monitoring tool
  - Comprehensive diagnostics and performance testing
  - Environment validation and troubleshooting

## Testing Tools

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

All Python scripts should be run from the project root directory:

### Document Management
```bash
# Upload documents to vector database
python tools/upload_docs.py

# Clear and rebuild vector database
python tools/clear_and_reload_docs.py

# Manage vector database (stats, backup, restore)
python tools/vector_db_manager.py stats
python tools/vector_db_manager.py backup --name my_backup
python tools/vector_db_manager.py restore --name my_backup

# Document management operations
python tools/manage_docs.py list
python tools/manage_docs.py search "your query"
python tools/manage_docs.py stats

# RAG system health check
python tools/rag_health_check.py
```

### Testing and Development
```bash
python tools/interactive_chat.py
python tools/debug_server.py
python tools/test_openai_key.py
python tools/quick_test.py
```

For the bash script:
```bash
chmod +x tools/test_api.sh
./tools/test_api.sh
```

## Note

These are development utilities and should not be used in production.
For unit tests, see the `tests/` directory.