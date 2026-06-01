"""Low-code monitor and mission builder API."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from api.action_execution import stage_api_action
from api.async_job_runner import enqueue_registered_job, enqueue_response
from api.exceptions import NotFoundError, ValidationError
from api.routers.auth import ActorDep
from ontology.runtime_read_service import OntologyRuntimeReadService

router = APIRouter()


class BuilderMutationOptions(BaseModel):
    reason: str | None = None
    apply: bool = False
    approval_note: str | None = None


class MonitorDefinitionRequest(BuilderMutationOptions):
    monitor_id: str | None = None
    name: str
    description: str | None = None
    template_id: str = "custom"
    scope: dict[str, Any] = Field(default_factory=dict)
    trigger_type: str = "custom"
    condition: str
    definition: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    source_requirements: list[dict[str, Any]] = Field(default_factory=list)
    cadence: dict[str, Any] = Field(default_factory=dict)
    severity: Literal["low", "medium", "high"] = "medium"
    output_policy: dict[str, Any] = Field(default_factory=dict)
    approval_behavior: Literal["hit_only_then_human_review"] = "hit_only_then_human_review"

    @field_validator("name", "condition")
    @classmethod
    def _required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("name and condition are required.")
        return text


class MissionDefinitionRequest(BuilderMutationOptions):
    mission_id: str | None = None
    name: str
    description: str | None = None
    template_id: str = "custom"
    mission_type: str = "monitor_review"
    scope: dict[str, Any] = Field(default_factory=dict)
    workflow_name: str | None = None
    schedule: dict[str, Any] = Field(default_factory=dict)
    source_requirements: list[dict[str, Any]] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    output_policy: dict[str, Any] = Field(default_factory=dict)
    approval_behavior: Literal["hit_only_then_human_review"] = "hit_only_then_human_review"

    @field_validator("name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("name is required.")
        return text


class RunBuilderRequest(BaseModel):
    monitor_id: str | None = None
    mission_id: str | None = None
    source: str = "manual"


class PreviewMonitorRequest(MonitorDefinitionRequest):
    pass


def _mutation_payload(body: BaseModel, *, exclude: set[str] | None = None) -> dict[str, Any]:
    return body.model_dump(exclude={"reason", "apply", "approval_note", *(exclude or set())}, exclude_none=True)


@router.get("/monitor-builder/definitions")
def list_definitions(status: str | None = None):
    reads = OntologyRuntimeReadService()
    return {
        "monitors": reads.monitor_definitions(status=status, limit=200),
        "missions": reads.mission_definitions(status=status, limit=200),
    }


@router.post("/monitor-builder/monitors")
def create_monitor(body: MonitorDefinitionRequest, actor: ActorDep):
    payload = _mutation_payload(body)
    return stage_api_action(
        "create_monitor_definition",
        payload,
        source_id="monitor_builder.create_monitor",
        actor=actor,
        reason=body.reason or f"Create monitor definition: {body.name}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.put("/monitor-builder/monitors/{monitor_id}")
def update_monitor(monitor_id: str, body: MonitorDefinitionRequest, actor: ActorDep):
    payload = {"monitor_id": monitor_id, **_mutation_payload(body, exclude={"monitor_id"})}
    return stage_api_action(
        "update_monitor_definition",
        payload,
        source_id=f"monitor_builder.update_monitor:{monitor_id}",
        actor=actor,
        reason=body.reason or f"Update monitor definition: {body.name}",
        apply=body.apply,
        approval_note=body.approval_note,
        entity_id=monitor_id,
    )


@router.post("/monitor-builder/monitors/{monitor_id}/disable")
def disable_monitor(monitor_id: str, actor: ActorDep, body: BuilderMutationOptions | None = None):
    return stage_api_action(
        "disable_monitor_definition",
        {"monitor_id": monitor_id},
        source_id=f"monitor_builder.disable_monitor:{monitor_id}",
        actor=actor,
        reason=(body.reason if body else None) or f"Disable monitor definition {monitor_id}",
        apply=body.apply if body else False,
        approval_note=body.approval_note if body else None,
        entity_id=monitor_id,
    )


@router.post("/monitor-builder/missions")
def create_mission(body: MissionDefinitionRequest, actor: ActorDep):
    payload = _mutation_payload(body)
    return stage_api_action(
        "create_mission_definition",
        payload,
        source_id="monitor_builder.create_mission",
        actor=actor,
        reason=body.reason or f"Create mission definition: {body.name}",
        apply=body.apply,
        approval_note=body.approval_note,
    )


@router.put("/monitor-builder/missions/{mission_id}")
def update_mission(mission_id: str, body: MissionDefinitionRequest, actor: ActorDep):
    payload = {"mission_id": mission_id, **_mutation_payload(body, exclude={"mission_id"})}
    return stage_api_action(
        "update_mission_definition",
        payload,
        source_id=f"monitor_builder.update_mission:{mission_id}",
        actor=actor,
        reason=body.reason or f"Update mission definition: {body.name}",
        apply=body.apply,
        approval_note=body.approval_note,
        entity_id=mission_id,
    )


@router.post("/monitor-builder/missions/{mission_id}/disable")
def disable_mission(mission_id: str, actor: ActorDep, body: BuilderMutationOptions | None = None):
    return stage_api_action(
        "disable_mission_definition",
        {"mission_id": mission_id},
        source_id=f"monitor_builder.disable_mission:{mission_id}",
        actor=actor,
        reason=(body.reason if body else None) or f"Disable mission definition {mission_id}",
        apply=body.apply if body else False,
        approval_note=body.approval_note if body else None,
        entity_id=mission_id,
    )


@router.post("/monitor-builder/preview")
def preview_monitor(body: PreviewMonitorRequest):
    from api.mission_runner import evaluate_monitor_definition

    try:
        return evaluate_monitor_definition(_mutation_payload(body))
    except Exception as exc:
        raise ValidationError(str(exc)) from exc


@router.post("/monitor-builder/run")
def run_builder_definitions(body: RunBuilderRequest | None = None):
    payload = body.model_dump(exclude_none=True) if body else {"source": "manual"}
    monitor_id = payload.get("monitor_id")
    mission_id = payload.get("mission_id")
    if monitor_id and not OntologyRuntimeReadService().get(str(monitor_id)):
        raise NotFoundError("MonitorDefinition", str(monitor_id))
    if mission_id and not OntologyRuntimeReadService().get(str(mission_id)):
        raise NotFoundError("MissionDefinition", str(mission_id))
    row, _disposition = enqueue_registered_job(
        "monitor_mission_runner",
        payload,
        cache_key=None,
        reuse_completed=False,
    )
    return enqueue_response(row, "/api/admin/jobs/{job_id}")
