"""API endpoints for platform adapters."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.adapters import (
    PlatformMessage,
    MessageType,
    # TeamsAdapter,
    # KakaoTalkAdapter,
    WebChatAdapter,
)
from app.agents.conversation import ConversationAgent
from app.api.v1.endpoints.auth import get_current_user


router = APIRouter()


class SendMessageRequest(BaseModel):
    """Request model for sending messages."""

    platform: str = Field(
        ..., description="Target platform (teams, kakaotalk, webchat)"
    )
    conversation_id: str = Field(..., description="Conversation ID")
    text: Optional[str] = Field(None, description="Message text")
    type: str = Field("text", description="Message type")
    attachments: Optional[List[Dict[str, Any]]] = Field(
        None, description="Message attachments"
    )
    buttons: Optional[List[Dict[str, str]]] = Field(
        None, description="Interactive buttons"
    )
    cards: Optional[List[Dict[str, Any]]] = Field(None, description="Rich cards")


class WebhookRequest(BaseModel):
    """Generic webhook request model."""

    platform: str = Field(..., description="Platform name")
    data: Dict[str, Any] = Field(..., description="Platform-specific webhook data")


# Initialize adapters (웹챗만 활성화)
adapters = {
    "webchat": WebChatAdapter(
        {
            "widget": {
                "script_url": "/static/moji-webchat.js",
                "api_url": "/api/v1/webchat",
                "theme": "light",
            }
        }
    ),
}

# Initialize conversation agent
conversation_agent = ConversationAgent()


@router.post("/send")
async def send_message(
    request: SendMessageRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Send a message to a specific platform."""
    if request.platform not in adapters:
        raise HTTPException(
            status_code=400, detail=f"Unsupported platform: {request.platform}"
        )

    adapter = adapters[request.platform]

    # Create platform message
    message = PlatformMessage(
        type=MessageType(request.type),
        text=request.text,
    )

    # Add conversation info
    message.conversation = await adapter.get_conversation_info(request.conversation_id)

    # Send message
    try:
        result = await adapter.send_message(message)
        return {
            "success": True,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/{platform}")
async def platform_webhook(
    platform: str, webhook_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle incoming webhooks from platforms."""
    if platform not in adapters:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")

    adapter = adapters[platform]

    try:
        # Convert to platform message
        message = await adapter.receive_message(webhook_data)

        # Process with conversation agent
        response = await conversation_agent.process_message(message)

        # Send response back through adapter
        if response:
            await adapter.send_message(response)

        return {"status": "processed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/webchat/ws")
async def webchat_websocket(websocket: WebSocket):
    """WebSocket endpoint for WebChat."""
    adapter = adapters["webchat"]

    # Attach conversation agent to adapter
    adapter.conversation_agent = conversation_agent

    try:
        # Handle WebSocket connection
        await adapter.handle_websocket(websocket)

    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        # Try to close websocket safely
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close(code=1000, reason=str(e))
        except Exception:
            # Ignore close errors (connection might already be closed)
            pass


@router.get("/webchat/page")
async def get_webchat_page():
    """Serve the full WebChat page (modular version)."""
    return FileResponse("app/static/moji-webchat-v2-modular.html")


@router.get("/webchat/widget")
async def get_webchat_widget() -> Dict[str, str]:
    """Get WebChat widget HTML."""
    adapter = adapters["webchat"]
    return {
        "html": adapter.get_widget_html(),
        "instructions": "Embed this HTML in your webpage to add MOJI WebChat.",
    }


@router.get("/platforms")
async def list_platforms() -> Dict[str, List[Dict[str, Any]]]:
    """List available platforms and their features."""
    platforms = []

    for name, adapter in adapters.items():
        platform_info = {
            "name": name,
            "features": {
                "buttons": adapter.supports_feature("buttons"),
                "cards": adapter.supports_feature("cards"),
                "files": adapter.supports_feature("files"),
                "images": adapter.supports_feature("images"),
                "audio": adapter.supports_feature("audio"),
                "video": adapter.supports_feature("video"),
                "location": adapter.supports_feature("location"),
                "typing_indicator": adapter.supports_feature("typing_indicator"),
            },
        }
        platforms.append(platform_info)

    return {"platforms": platforms}


@router.get("/platforms/{platform}/status")
async def platform_status(platform: str) -> Dict[str, Any]:
    """Get platform adapter status."""
    if platform not in adapters:
        raise HTTPException(status_code=404, detail=f"Platform not found: {platform}")

    adapter = adapters[platform]

    # Get platform-specific status
    status = {
        "platform": platform,
        "connected": True,  # Simplified for now
        "features": {
            feature: adapter.supports_feature(feature)
            for feature in ["buttons", "cards", "files", "images", "audio", "video"]
        },
    }

    if platform == "webchat":
        status["active_connections"] = len(adapter.connections)

    return status


# Teams-specific endpoints (주석 처리)
# @router.post("/teams/activity")
# async def teams_activity(activity: Dict[str, Any]) -> Dict[str, Any]:
#     """Handle Teams Bot Framework activities."""
#     return await platform_webhook("teams", activity)


# KakaoTalk-specific endpoints (주석 처리)
# @router.post("/kakao/message")
# async def kakao_message(message: Dict[str, Any]) -> Dict[str, Any]:
#     """Handle KakaoTalk messages."""
#     return await platform_webhook("kakaotalk", message)
