#!/bin/bash

# MOJI AI Agent API Test Script
# This script tests the main API endpoints

BASE_URL=${MOJI_BASE_URL:-http://localhost:8000}
USERNAME=${TEST_USERNAME:-testuser}
PASSWORD=${TEST_PASSWORD:-testpass123}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 MOJI AI Agent API Test Script"
echo "================================="
echo "Base URL: $BASE_URL"
echo ""

# Function to check if server is running
check_server() {
    echo -n "Checking server health... "
    if curl -s -f "$BASE_URL/api/v1/health" > /dev/null; then
        echo -e "${GREEN}✓ Server is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Server is not running${NC}"
        echo "Please start the server with: docker-compose up -d"
        return 1
    fi
}

# Function to register user
register_user() {
    echo -n "Registering user '$USERNAME'... "
    RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/auth/register" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}")
    
    if [[ $RESPONSE == *"username"* ]] || [[ $RESPONSE == *"already registered"* ]]; then
        echo -e "${GREEN}✓ User ready${NC}"
        return 0
    else
        echo -e "${RED}✗ Registration failed${NC}"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Function to get auth token
get_token() {
    echo -n "Getting auth token... "
    RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/auth/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=$USERNAME&password=$PASSWORD")
    
    TOKEN=$(echo $RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
    
    if [ -n "$TOKEN" ]; then
        echo -e "${GREEN}✓ Token obtained${NC}"
        echo "Token: ${TOKEN:0:20}..."
        return 0
    else
        echo -e "${RED}✗ Failed to get token${NC}"
        echo "Response: $RESPONSE"
        return 1
    fi
}

# Function to test chat
test_chat() {
    echo ""
    echo "Testing chat endpoint..."
    echo "------------------------"
    
    CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/chat/completions" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "user", "content": "Hello! What is 2+2?"}
            ]
        }')
    
    if [[ $CHAT_RESPONSE == *"choices"* ]]; then
        echo -e "${GREEN}✓ Chat endpoint working${NC}"
        echo "Response preview:"
        echo "$CHAT_RESPONSE" | grep -o '"content":"[^"]*' | head -1
    else
        echo -e "${RED}✗ Chat endpoint failed${NC}"
        echo "Response: $CHAT_RESPONSE"
    fi
}

# Function to test LLM status
test_llm_status() {
    echo ""
    echo "Checking LLM status..."
    echo "---------------------"
    
    LLM_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/llm/status" \
        -H "Authorization: Bearer $TOKEN")
    
    if [[ $LLM_RESPONSE == *"provider"* ]]; then
        echo -e "${GREEN}✓ LLM status endpoint working${NC}"
        echo "LLM Info:"
        echo "$LLM_RESPONSE" | grep -o '"provider":"[^"]*\|"model":"[^"]*'
    else
        echo -e "${YELLOW}⚠ LLM status endpoint not available${NC}"
    fi
}

# Function to test RAG upload
test_rag_upload() {
    echo ""
    echo "Testing RAG document upload..."
    echo "------------------------------"
    
    # Create a test file
    echo "This is a test document for MOJI AI Agent." > test_doc.txt
    
    RAG_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/rag/documents/upload" \
        -H "Authorization: Bearer $TOKEN" \
        -F "file=@test_doc.txt")
    
    if [[ $RAG_RESPONSE == *"processed"* ]] || [[ $RAG_RESPONSE == *"success"* ]]; then
        echo -e "${GREEN}✓ RAG upload endpoint working${NC}"
    else
        echo -e "${YELLOW}⚠ RAG upload endpoint issue${NC}"
        echo "Response: $RAG_RESPONSE"
    fi
    
    rm -f test_doc.txt
}

# Function to test vector store
test_vector_store() {
    echo ""
    echo "Testing vector store..."
    echo "----------------------"
    
    # List vector stores
    VECTOR_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/vectorstore/stores" \
        -H "Authorization: Bearer $TOKEN")
    
    if [[ $VECTOR_RESPONSE == *"total_stores"* ]] || [[ $VECTOR_RESPONSE == *"default_store"* ]]; then
        echo -e "${GREEN}✓ Vector store endpoint working${NC}"
        echo "Vector stores: $VECTOR_RESPONSE" | head -1
    else
        echo -e "${YELLOW}⚠ Vector store endpoint issue${NC}"
    fi
}

# Function to test agents
test_agents() {
    echo ""
    echo "Testing agents endpoint..."
    echo "-------------------------"
    
    AGENTS_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/agents" \
        -H "Authorization: Bearer $TOKEN")
    
    if [[ $AGENTS_RESPONSE == *"agents"* ]] || [[ $AGENTS_RESPONSE == *"chat_agent"* ]]; then
        echo -e "${GREEN}✓ Agents endpoint working${NC}"
        echo "Available agents:"
        echo "$AGENTS_RESPONSE" | grep -o '"[^"]*_agent"' | sort -u
    else
        echo -e "${YELLOW}⚠ Agents endpoint issue${NC}"
    fi
}

# Main test flow
main() {
    check_server || exit 1
    echo ""
    
    register_user || exit 1
    get_token || exit 1
    
    test_chat
    test_llm_status
    test_agents
    test_rag_upload
    test_vector_store
    
    echo ""
    echo "================================="
    echo -e "${GREEN}✓ API tests completed!${NC}"
    echo ""
    echo "To test interactively, run:"
    echo "  python test_chat.py"
}

# Run main function
main