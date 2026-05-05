"""Lightweight request models shared by agent chat routes and workers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

ChatText = Annotated[str, Field(min_length=1, max_length=64 * 1024)]
ScreenShortText = Annotated[str, Field(max_length=512)]
ScreenValueText = Annotated[str, Field(max_length=4096)]
ToolNameText = Annotated[str, Field(max_length=128)]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: ChatText


class ScreenContextModel(BaseModel):
    page_name: ScreenShortText
    route: ScreenShortText
    ticker: ScreenShortText | None = None
    metrics: dict[ScreenShortText, ScreenValueText] | None = Field(default=None, max_length=100)
    filters: dict[ScreenShortText, ScreenValueText] | None = Field(default=None, max_length=100)
    summary: ScreenValueText | None = None
    corresponding_tools: list[ToolNameText] | None = Field(default=None, max_length=50)


PreferenceLevel = Literal["less", "balanced", "more"]
Personality = Literal["friendly", "pragmatic"]
CustomInstructionText = Annotated[str, Field(max_length=2000)]


class AgentResponsePreferences(BaseModel):
    personality: Personality = "pragmatic"
    warmth: PreferenceLevel = "less"
    enthusiasm: PreferenceLevel = "less"
    headers_lists: PreferenceLevel = "less"
    emoji: PreferenceLevel = "less"
    fast_answers: bool = True
    thinking_enabled: bool = False
    custom_instructions: CustomInstructionText | None = None


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    screen_context: ScreenContextModel | None = None
    response_preferences: AgentResponsePreferences | None = None


class AgentChatRequestV2(BaseModel):
    """V2 request: frontend sends only the new message + session ID."""

    session_id: ScreenShortText | None = None
    client_turn_id: ScreenShortText | None = None
    message: ChatText
    screen_context: ScreenContextModel | None = None
    response_preferences: AgentResponsePreferences | None = None
    finalize_synchronously: bool = False


class AgentChatJobRequest(AgentChatRequestV2):
    """Payload executed by the durable async agent worker."""

    actor: dict[str, Any] | None = None
    message_count: int | None = None
