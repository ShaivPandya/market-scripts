"""Lightweight request models shared by agent chat routes and workers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    """Frontend sends only the new message plus the server-managed session ID."""

    session_id: ScreenShortText | None = None
    client_turn_id: ScreenShortText | None = None
    message: ChatText
    screen_context: ScreenContextModel | None = None
    response_preferences: AgentResponsePreferences | None = None
    finalize_synchronously: bool = False
    allow_workflow_handoff: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_messages(cls, data: Any) -> Any:
        if not isinstance(data, dict) or data.get("message"):
            return data
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data
        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user" and item.get("content"):
                updated = dict(data)
                updated["message"] = item["content"]
                return updated
        return data


class AgentChatJobRequest(AgentChatRequest):
    """Payload executed by the durable async agent worker."""

    actor: dict[str, Any] | None = None
    message_count: int | None = None


FeedbackDecision = Literal["approve", "reject", "correct"]
FeedbackTag = Literal["routing", "tools", "source_quality", "synthesis", "calibration", "policy_boundary"]
FeedbackNoteText = Annotated[str, Field(max_length=4000)]
CorrectedResponseText = Annotated[str, Field(max_length=64 * 1024)]


class AgentResponseFeedbackRequest(BaseModel):
    """Submit or update explicit human feedback for one completed agent response."""

    trajectory_id: ScreenShortText | None = None
    session_id: ScreenShortText | None = None
    client_turn_id: ScreenShortText | None = None
    decision: FeedbackDecision
    corrected_response: CorrectedResponseText | None = None
    failure_tags: list[FeedbackTag] = Field(default_factory=list, max_length=12)
    notes: FeedbackNoteText | None = None
    eligible_for_training: bool = False

    @model_validator(mode="after")
    def _requires_target(self) -> AgentResponseFeedbackRequest:
        if self.trajectory_id or (self.session_id and self.client_turn_id):
            return self
        raise ValueError("Provide trajectory_id or both session_id and client_turn_id")
