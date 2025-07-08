"""Web Chat adapter implementation with WebSocket support."""

import asyncio
import uuid
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from .base import (
    BaseAdapter,
    PlatformMessage,
    MessageType,
    User,
    Conversation,
    Attachment,
    AttachmentType,
)


class WebChatConnection:
    """Represents a WebChat connection."""

    def __init__(self, websocket: WebSocket, user_id: str, session_id: str):
        self.websocket = websocket
        self.user_id = user_id
        self.session_id = session_id
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

    async def send(self, data: Dict[str, Any]) -> None:
        """Send data through WebSocket."""
        await self.websocket.send_json(data)
        self.last_activity = datetime.utcnow()


class WebChatAdapter(BaseAdapter):
    """Web Chat adapter for embedded widget."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize WebChat adapter."""
        super().__init__(config)
        self.connections: Dict[str, WebChatConnection] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.widget_config = config.get("widget", {})
        self._cleanup_task: Optional[asyncio.Task] = None
        self.conversation_agent = None  # Will be set by the router

    async def connect(self) -> None:
        """Start WebChat service."""
        # Start connection cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())

    async def disconnect(self) -> None:
        """Stop WebChat service."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn in list(self.connections.values()):
            await conn.websocket.close()

        self.connections.clear()
        self.conversations.clear()

    async def handle_websocket(
        self, websocket: WebSocket, session_id: Optional[str] = None
    ) -> None:
        """Handle incoming WebSocket connection."""
        await websocket.accept()
        print("[WebChat] WebSocket connection accepted")

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"[WebChat] Generated session ID: {session_id}")

        # Wait for authentication message with timeout
        try:
            print("[WebChat] Waiting for authentication message...")
            auth_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            print(f"[WebChat] Received auth data: {auth_data}")
            
            user_id = auth_data.get("user_id", f"guest_{uuid.uuid4().hex[:8]}")
            user_name = auth_data.get("user_name", "Guest User")
            print(f"[WebChat] Authenticated user: {user_id} ({user_name})")
        except asyncio.TimeoutError:
            print("[WebChat] Authentication timeout, using guest credentials")
            user_id = f"guest_{uuid.uuid4().hex[:8]}"
            user_name = "Guest User"
        except Exception as e:
            print(f"[WebChat] Authentication error: {e}, using guest credentials")
            user_id = f"guest_{uuid.uuid4().hex[:8]}"
            user_name = "Guest User"

        # Create connection
        connection = WebChatConnection(websocket, user_id, session_id)
        self.connections[session_id] = connection
        print(f"[WebChat] Connection created for session: {session_id}")

        # Create or get conversation
        if session_id not in self.conversations:
            self.conversations[session_id] = Conversation(
                id=session_id,
                platform="webchat",
                type="direct",
                name=f"Chat with {user_name}",
                metadata={"started_at": datetime.utcnow().isoformat()},
            )
            print(f"[WebChat] Conversation created for session: {session_id}")

        # Send welcome message
        await self._send_system_message(
            connection, "Welcome to MOJI Web Chat! How can I help you today?"
        )

        try:
            # Handle incoming messages with improved stability
            while True:
                try:
                    # WebSocket 메시지 수신 (타임아웃 추가)
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
                    
                    # Handle ping/pong messages
                    if data.get("type") == "ping":
                        await connection.send({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                        connection.last_activity = datetime.utcnow()
                        continue
                    elif data.get("type") == "pong":
                        # Update last activity for pong messages
                        connection.last_activity = datetime.utcnow()
                        continue
                    
                except asyncio.TimeoutError:
                    # 타임아웃 시 ping 전송으로 연결 확인
                    try:
                        await connection.send({"type": "ping", "timestamp": datetime.utcnow().isoformat()})
                        continue
                    except Exception:
                        print("[WebChat] Connection timeout, closing...")
                        break
                except Exception as e:
                    print(f"[WebChat] WebSocket receive error: {e}")
                    break

                message = await self._process_incoming_message(data, connection)

                # Process message with LLM router
                if message.text and message.text.strip():
                    try:
                        # Check for RAG commands
                        if message.text.startswith("/rag "):
                            # Handle RAG queries
                            from app.rag.enhanced_rag import rag_pipeline

                            query = message.text[5:].strip()  # Remove "/rag " prefix
                            print(f"[WebChat] RAG query: {query}")

                            # Get answer with confidence
                            result = await rag_pipeline.answer_with_confidence(query)

                            # Format response
                            response_text = f"**답변**: {result['answer']}\n\n"
                            response_text += f"**신뢰도**: {result['confidence']}\n"
                            response_text += f"**근거**: {result['reasoning']}\n"

                            if result["sources"]:
                                response_text += "\n**출처**:\n"
                                for source in result["sources"]:
                                    response_text += f"- {source}\n"

                            await connection.send(
                                {
                                    "id": str(uuid.uuid4()),
                                    "type": "text",
                                    "text": response_text,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )

                        elif message.text == "/rag-stats":
                            # Show RAG statistics
                            from app.rag.enhanced_rag import rag_pipeline

                            stats = rag_pipeline.get_collection_stats()
                            response_text = "**RAG 시스템 통계**\n"
                            response_text += (
                                f"- 총 문서 수: {stats.get('total_documents', 0)}\n"
                            )
                            response_text += (
                                f"- 청크 크기: {stats.get('chunk_size', 0)}\n"
                            )
                            response_text += (
                                f"- 청크 오버랩: {stats.get('chunk_overlap', 0)}\n"
                            )
                            response_text += f"- 임베딩 모델: {stats.get('embedding_model', 'Unknown')}\n"

                            await connection.send(
                                {
                                    "id": str(uuid.uuid4()),
                                    "type": "text",
                                    "text": response_text,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )

                        elif message.text == "/rag-help":
                            # Show RAG help
                            help_text = """**RAG 시스템 사용법**

1. **문서 업로드**: API를 통해 문서를 업로드하세요
   - 지원 형식: .txt, .md, .docx

2. **질문하기**: `/rag [질문]`
   예시: `/rag 프로젝트의 주요 목표는 무엇인가요?`

3. **통계 보기**: `/rag-stats`
   현재 저장된 문서 수와 설정을 확인합니다.

4. **일반 대화**: 일반 메시지는 ChatGPT가 답변합니다.

**특징**:
- 질의 재작성으로 검색 정확도 향상
- 신뢰도 점수 제공 (HIGH/MEDIUM/LOW)
- 답변 근거 및 출처 표시"""

                            await connection.send(
                                {
                                    "id": str(uuid.uuid4()),
                                    "type": "text",
                                    "text": help_text,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )

                        else:
                            # Initialize components on first use (lazy loading)
                            from app.rag.enhanced_rag import rag_pipeline
                            from app.llm.router import llm_router
                            from app.agents.manager import agent_manager

                            # Initialize LLM router if not already done
                            if not llm_router.current_provider:
                                await llm_router.initialize()

                            # Initialize agent system if not already done
                            if not agent_manager.agents:
                                await agent_manager.initialize_default_agents()

                            # Log incoming message
                            print(f"[WebChat] Received message: {message.text}")

                            # Check RAG setting from client
                            use_rag = data.get(
                                "useRag", True
                            )  # Default to True for backward compatibility
                            strict_rag_mode = data.get(
                                "strictRagMode", True
                            )  # 엄격한 RAG 모드 (문서 기반 답변만)
                            provider = data.get("provider")
                            model = data.get("model")

                            print(
                                f"[WebChat] RAG setting: {use_rag}, Strict mode: {strict_rag_mode}, Provider: {provider}, Model: {model}"
                            )

                            try:
                                if use_rag:
                                    # Try Hybrid RAG first (improved search)
                                    print("[WebChat] Trying Hybrid RAG search...")
                                    from app.rag.enhanced_rag import get_hybrid_pipeline

                                    hybrid_pipeline = get_hybrid_pipeline()

                                    # RAG 검색 임계값을 엄격 모드에 따라 조정
                                    search_threshold = 0.3 if strict_rag_mode else 0.1

                                    # Context에 strict mode 정보 포함
                                    search_context = {
                                        "strict_mode": strict_rag_mode,
                                        "provider": provider,
                                        "model": model,
                                    }
                                    
                                    rag_result = await hybrid_pipeline.answer_with_hybrid_search(
                                        message.text,
                                        k=5,
                                        score_threshold=search_threshold,
                                        context=search_context,
                                    )

                                    # RAG 결과 타입 체크 추가
                                    if (
                                        rag_result
                                        and isinstance(rag_result, dict)  # 타입 체크 추가
                                        and rag_result.get("sources")
                                        and len(rag_result.get("sources", [])) > 0
                                        and rag_result.get("answer")
                                        and "관련된 정보를 찾을 수 없습니다"
                                        not in rag_result.get("answer", "")
                                    ):

                                        print("[WebChat] Using Hybrid RAG response")
                                        # Use Hybrid RAG response
                                        response_text = rag_result["answer"]

                                        # Enhanced metadata with hybrid search info
                                        sources = rag_result.get("sources", [])
                                        confidence = rag_result.get(
                                            "confidence", "MEDIUM"
                                        )
                                        search_metadata = rag_result.get(
                                            "search_metadata", {}
                                        )

                                        # sources가 리스트가 아닌 경우 안전하게 처리
                                        if not isinstance(sources, list):
                                            sources = []
                                            
                                        metadata = {
                                            "mode": "Hybrid-RAG",
                                            "confidence": confidence,
                                            "sources": len(sources),
                                            "search_type": search_metadata.get(
                                                "search_type", "hybrid"
                                            ),
                                            "total_results": search_metadata.get(
                                                "total_results", 0
                                            ),
                                        }
                                        
                                        # RAG 테스트를 위한 출처 정보 표시
                                        search_info = f"신뢰도: {confidence}, 출처: {len(sources)}개"
                                        if search_metadata.get("total_results"):
                                            search_info += f", 검색결과: {search_metadata['total_results']}개"
                                        
                                        # 출처 문서 이름들 추가 (안전한 처리)
                                        source_files = []
                                        for source in sources:
                                            if isinstance(source, dict):
                                                filename = source.get('filename', source.get('source', '알 수 없음'))
                                                source_files.append(filename)
                                            elif isinstance(source, str):
                                                source_files.append(source)
                                            else:
                                                source_files.append('알 수 없음')
                                        
                                        source_files = list(set(source_files))  # 중복 제거
                                        source_files_str = ", ".join(source_files[:3])  # 최대 3개만 표시
                                        if len(source_files) > 3:
                                            source_files_str += f" 외 {len(source_files)-3}개"
                                        
                                        response_text += f"\n\n📊 **RAG 정보**: {search_info}\n📄 **출처**: {source_files_str}"

                                        await connection.send(
                                            {
                                                "id": str(uuid.uuid4()),
                                                "type": "text",
                                                "text": response_text,
                                                "timestamp": datetime.utcnow().isoformat(),
                                                "metadata": metadata,
                                            }
                                        )
                                    else:
                                        # No relevant documents found
                                        if strict_rag_mode:
                                            # 엄격 모드: LLM 폴백 없이 문서 기반 답변만 제공
                                            print(
                                                "[WebChat] No relevant documents found, strict mode - no LLM fallback"
                                            )
                                            await connection.send(
                                                {
                                                    "id": str(uuid.uuid4()),
                                                    "type": "text",
                                                    "text": "죄송합니다. 업로드된 문서에서 관련 정보를 찾을 수 없습니다. 다른 질문을 해주시거나 관련 문서를 업로드해 주세요.",
                                                    "timestamp": datetime.utcnow().isoformat(),
                                                    "metadata": {
                                                        "mode": "Strict-RAG",
                                                        "confidence": "LOW",
                                                        "sources": 0,
                                                        "message": "문서 기반 답변만 제공 (LLM 폴백 비활성화)",
                                                    },
                                                }
                                            )
                                        else:
                                            # 일반 모드: LLM 폴백 사용
                                            print(
                                                "[WebChat] No relevant documents found, using regular LLM"
                                            )
                                            await self._generate_llm_response(
                                                connection,
                                                message.text,
                                                provider,
                                                model,
                                                "Hybrid",
                                            )
                                else:
                                    # RAG disabled, use LLM directly
                                    print("[WebChat] RAG disabled, using LLM directly")
                                    await self._generate_llm_response(
                                        connection, message.text, provider, model, "LLM"
                                    )

                            except Exception as rag_error:
                                error_msg = str(rag_error)
                                print(f"[WebChat] RAG error: {error_msg}, falling back to LLM")
                                
                                # 구체적인 에러 메시지 제공
                                if "'NoneType' object has no attribute '_collection'" in error_msg:
                                    error_detail = "벡터 데이터베이스가 초기화되지 않았습니다. 문서를 먼저 업로드해주세요."
                                elif "No documents found" in error_msg:
                                    error_detail = "업로드된 문서가 없습니다. RAG 기능을 사용하려면 문서를 먼저 업로드해주세요."
                                elif "unexpected keyword argument" in error_msg:
                                    error_detail = "RAG 시스템 설정 오류가 발생했습니다."
                                else:
                                    error_detail = f"RAG 처리 중 오류: {error_msg}"
                                
                                print(f"[WebChat] Error detail: {error_detail}")
                                
                                # Fallback to LLM with error context
                                await self._generate_llm_response(
                                    connection,
                                    message.text,
                                    provider,
                                    model,
                                    f"RAG-오류-폴백 ({error_detail})",
                                )

                    except Exception as e:
                        print(f"[WebChat] Error processing message: {str(e)}")
                        import traceback

                        traceback.print_exc()
                        await self.handle_error(e)
                        # Send error message to client
                        await connection.send(
                            {
                                "id": str(uuid.uuid4()),
                                "type": "system",
                                "text": f"죄송합니다. 메시지 처리 중 오류가 발생했습니다: {str(e)}",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

        except WebSocketDisconnect:
            # Clean up connection
            if session_id in self.connections:
                del self.connections[session_id]
        except Exception as e:
            await self.handle_error(e)
            # Clean up connection
            if session_id in self.connections:
                del self.connections[session_id]
            # Try to close websocket safely
            try:
                if not websocket.client_state.DISCONNECTED:
                    await websocket.close(code=1000, reason="Server error")
            except Exception:
                # Ignore close errors (connection might already be closed)
                pass

    async def send_message(self, message: PlatformMessage) -> Dict[str, Any]:
        """Send message to WebChat client."""
        if not await self.validate_message(message):
            raise ValueError("Invalid message format")

        # Find target connection
        target_session = message.conversation.id if message.conversation else None
        if not target_session or target_session not in self.connections:
            return {
                "id": str(message.id),
                "status": "failed",
                "error": "Connection not found",
            }

        connection = self.connections[target_session]

        # Convert to WebChat format
        webchat_message = await self._convert_to_webchat_format(message)

        # Send through WebSocket
        await connection.send(webchat_message)

        return {
            "id": str(message.id),
            "status": "sent",
            "platform": "webchat",
            "timestamp": message.timestamp.isoformat(),
        }

    async def receive_message(self, raw_message: Dict[str, Any]) -> PlatformMessage:
        """Convert WebChat message to platform message."""
        msg = PlatformMessage(
            id=raw_message.get("id", str(uuid.uuid4())),
            type=MessageType(raw_message.get("type", "text")),
            text=raw_message.get("text"),
            timestamp=datetime.fromisoformat(
                raw_message.get("timestamp", datetime.utcnow().isoformat())
            ),
            platform_specific={"webchat_data": raw_message},
        )

        # Set user from raw message
        if raw_message.get("user"):
            msg.user = User(
                id=raw_message["user"]["id"],
                name=raw_message["user"]["name"],
                platform="webchat",
                avatar_url=raw_message["user"].get("avatar"),
            )

        # Set conversation
        if raw_message.get("session_id"):
            msg.conversation = self.conversations.get(raw_message["session_id"])

        # Process attachments
        if raw_message.get("attachments"):
            msg.attachments = [
                Attachment(
                    type=AttachmentType(att["type"]),
                    url=att.get("url"),
                    file_name=att.get("name"),
                    file_size=att.get("size"),
                    mime_type=att.get("mime_type"),
                )
                for att in raw_message["attachments"]
            ]

        return msg

    async def get_user_info(self, user_id: str) -> User:
        """Get WebChat user information."""
        # Find user in active connections
        for conn in self.connections.values():
            if conn.user_id == user_id:
                return User(
                    id=user_id,
                    name=f"User {user_id}",
                    platform="webchat",
                    metadata={
                        "session_id": conn.session_id,
                        "connected_at": conn.connected_at.isoformat(),
                    },
                )

        # Return default user info
        return User(
            id=user_id,
            name=f"User {user_id}",
            platform="webchat",
        )

    async def get_conversation_info(self, conversation_id: str) -> Conversation:
        """Get WebChat conversation information."""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]

        return Conversation(
            id=conversation_id,
            platform="webchat",
            type="direct",
        )

    async def broadcast_message(self, message: PlatformMessage) -> Dict[str, Any]:
        """Broadcast message to all connected clients."""
        webchat_message = await self._convert_to_webchat_format(message)

        # Send to all connections
        results = []
        for session_id, connection in self.connections.items():
            try:
                await connection.send(webchat_message)
                results.append(session_id)
            except Exception as e:
                await self.handle_error(e)

        return {
            "status": "broadcast",
            "recipients": len(results),
            "sessions": results,
        }

    async def _generate_llm_response(
        self,
        connection: WebChatConnection,
        text: str,
        provider: str = None,
        model: str = None,
        mode: str = "LLM",
    ) -> None:
        """Generate LLM response and send to client"""
        from app.llm.router import llm_router
        from langchain_core.messages import HumanMessage

        try:
            # Create messages list for LLM
            messages = [HumanMessage(content=text)]

            # Generate response with optional provider/model
            print(
                f"[WebChat] Generating LLM response with provider={provider}, model={model}..."
            )
            response = await llm_router.generate(
                messages=messages, provider=provider, model=model
            )
            print(f"[WebChat] Generated response: {response.content}")

            # Add mode indicator to response (제거됨)
            response_text = response.content
            # if mode == "LLM":
            #     response_text += f"\n\n🤖 **LLM 응답** ({provider or 'default'})"
            # elif mode == "Hybrid":
            #     response_text += "\n\n🔄 **하이브리드 응답** (RAG → LLM 폴백)"
            # elif mode == "Error-Fallback":
            #     response_text += "\n\n⚠️ **오류 복구** (LLM 폴백)"

            # Send response back to client
            await connection.send(
                {
                    "id": str(uuid.uuid4()),
                    "type": "text",
                    "text": response_text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {"mode": mode, "provider": provider, "model": model},
                }
            )

        except Exception as e:
            print(f"[WebChat] LLM generation error: {str(e)}")
            await connection.send(
                {
                    "id": str(uuid.uuid4()),
                    "type": "text",
                    "text": f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    async def _process_incoming_message(
        self, data: Dict[str, Any], connection: WebChatConnection
    ) -> PlatformMessage:
        """Process incoming WebChat message."""
        # Add metadata
        data["user"] = {
            "id": connection.user_id,
            "name": f"User {connection.user_id}",
        }
        data["session_id"] = connection.session_id
        data["timestamp"] = datetime.utcnow().isoformat()

        return await self.receive_message(data)

    async def _convert_to_webchat_format(
        self, message: PlatformMessage
    ) -> Dict[str, Any]:
        """Convert platform message to WebChat format."""
        data = {
            "id": str(message.id),
            "type": message.type.value,
            "timestamp": message.timestamp.isoformat(),
        }

        if message.text:
            data["text"] = message.text

        if message.attachments:
            data["attachments"] = [
                {
                    "type": att.type.value,
                    "url": att.url,
                    "name": att.file_name,
                    "size": att.file_size,
                    "mime_type": att.mime_type,
                }
                for att in message.attachments
            ]

        if message.buttons:
            data["buttons"] = [
                {
                    "text": btn.text,
                    "value": btn.value,
                    "action": btn.action,
                }
                for btn in message.buttons
            ]

        if message.cards:
            data["cards"] = [
                {
                    "title": card.title,
                    "subtitle": card.subtitle,
                    "text": card.text,
                    "image": card.image_url,
                    "buttons": [
                        {"text": btn.text, "value": btn.value} for btn in card.buttons
                    ],
                }
                for card in message.cards
            ]

        return data

    async def _send_system_message(
        self, connection: WebChatConnection, text: str
    ) -> None:
        """Send system message to client."""
        await connection.send(
            {
                "id": str(uuid.uuid4()),
                "type": "system",
                "text": text,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def _cleanup_inactive_connections(self) -> None:
        """Clean up inactive connections periodically."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                now = datetime.utcnow()
                inactive_threshold = 3600  # 1 hour

                for session_id, conn in list(self.connections.items()):
                    if (now - conn.last_activity).total_seconds() > inactive_threshold:
                        await conn.websocket.close()
                        del self.connections[session_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.handle_error(e)

    def get_widget_html(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate HTML for embeddable widget."""
        widget_config = config or self.widget_config

        return f"""
        <!-- MOJI Web Chat Widget -->
        <div id="moji-webchat-widget"></div>
        <script src="{widget_config.get('script_url', '/static/moji-webchat.js')}"></script>
        <script>
          MojiWebChat.init({{
            apiUrl: '{widget_config.get('api_url', '/api/v1/webchat')}',
            theme: '{widget_config.get('theme', 'light')}',
            position: '{widget_config.get('position', 'bottom-right')}',
            title: '{widget_config.get('title', 'MOJI Assistant')}',
            placeholder: '{widget_config.get('placeholder', 'Type your message...')}',
          }});
        </script>
        """

    def supports_feature(self, feature: str) -> bool:
        """Check WebChat feature support."""
        features = {
            "buttons": True,
            "cards": True,
            "files": True,
            "images": True,
            "audio": True,
            "video": True,
            "location": False,
            "typing_indicator": True,
            "read_receipts": True,
            "reactions": False,
            "persistent_history": True,
            "file_upload": True,
        }
        return features.get(feature, False)
