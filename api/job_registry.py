"""Registry for durable async job types."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JobSpec:
    job_type: str
    request_model: str | None
    compute_func: str
    cache_key_func: str | None
    queue_name: str = "default"
    timeout_s: int = 180
    completed_ttl_s: int = 24 * 60 * 60
    failed_ttl_s: int = 7 * 24 * 60 * 60
    stale_grace_s: int | None = None
    error_message: str = "Job failed"
    supports_progress: bool = False
    initial_progress: dict[str, Any] | None = None


def _env_int(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_queue(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


DEFAULT_COMPLETED_TTL_S = _env_int("ASYNC_JOB_COMPLETED_TTL_SECONDS", 24 * 60 * 60)
DEFAULT_FAILED_TTL_S = _env_int("ASYNC_JOB_FAILED_TTL_SECONDS", 7 * 24 * 60 * 60)


def import_string(path: str) -> Any:
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def parse_request(spec: JobSpec, payload: dict[str, Any]) -> Any:
    if spec.request_model is None:
        return payload
    model = import_string(spec.request_model)
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model(**payload)


def cache_key_for_payload(spec: JobSpec, payload: dict[str, Any]) -> str | None:
    if spec.cache_key_func is None:
        return None
    req = parse_request(spec, payload)
    func: Callable[[Any], str] = import_string(spec.cache_key_func)
    return func(req)


JOB_SPECS: dict[str, JobSpec] = {
    "analyzer": JobSpec(
        job_type="analyzer",
        request_model="api.routers.analyzer.AnalyzerRequest",
        compute_func="api.routers.analyzer._compute_analyzer_result",
        cache_key_func="api.routers.analyzer._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_ANALYZER", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_ANALYZER_SECONDS", 10 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Portfolio analyzer failed",
    ),
    "hedging": JobSpec(
        job_type="hedging",
        request_model="api.routers.hedging.HedgingRequest",
        compute_func="api.routers.hedging._compute_hedging_result",
        cache_key_func="api.routers.hedging._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_HEDGING", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_HEDGING_SECONDS", 180),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Hedging tool failed",
    ),
    "agent_chat_turn": JobSpec(
        job_type="agent_chat_turn",
        request_model="api.agent_models.AgentChatJobRequest",
        compute_func="api.agent_chat_worker._run_agent_chat_turn_job",
        cache_key_func="api.routers.agent._agent_chat_job_cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_AGENT", "agent"),
        timeout_s=_env_int("ASYNC_TIMEOUT_AGENT_CHAT_SECONDS", 20 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        stale_grace_s=_env_int("ASYNC_STALE_GRACE_AGENT_CHAT_SECONDS", 60),
        error_message="Agent chat turn failed",
    ),
    "sizer": JobSpec(
        job_type="sizer",
        request_model="api.routers.sizer.SizerRequest",
        compute_func="api.routers.sizer._compute_sizer_result",
        cache_key_func="api.routers.sizer._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_SIZER", "sizer"),
        timeout_s=_env_int("ASYNC_TIMEOUT_SIZER_SECONDS", 30),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        stale_grace_s=_env_int("ASYNC_STALE_GRACE_SIZER_SECONDS", 15),
        error_message="Portfolio sizer failed",
    ),
    "idea_evaluation": JobSpec(
        job_type="idea_evaluation",
        request_model="api.routers.ideas.IdeaEvaluationRequest",
        compute_func="api.routers.ideas._compute_idea_evaluation_result",
        cache_key_func="api.routers.ideas._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_IDEA_EVALUATION", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_IDEA_EVALUATION_SECONDS", 20 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        stale_grace_s=_env_int("ASYNC_STALE_GRACE_IDEA_EVALUATION_SECONDS", 60),
        error_message="Idea evaluation failed",
        supports_progress=True,
        initial_progress={"phase": "queued", "done": 0, "total": 1},
    ),
    "idea_comparison_evaluation": JobSpec(
        job_type="idea_comparison_evaluation",
        request_model="api.routers.ideas.IdeaComparisonEvaluationRequest",
        compute_func="api.routers.ideas._compute_idea_comparison_evaluation_result",
        cache_key_func=None,
        queue_name=_env_queue("ASYNC_QUEUE_IDEA_EVALUATION", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_IDEA_COMPARISON_EVALUATION_SECONDS", 45 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        stale_grace_s=_env_int("ASYNC_STALE_GRACE_IDEA_EVALUATION_SECONDS", 60),
        error_message="Idea comparison evaluation failed",
        supports_progress=True,
        initial_progress={"phase": "queued", "done": 0, "total": 1},
    ),
    "ontology": JobSpec(
        job_type="ontology",
        request_model="api.routers.ontology.OntologyQueryJobRequest",
        compute_func="api.routers.ontology._execute_query",
        cache_key_func="api.routers.ontology._job_cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_ONTOLOGY", "ontology"),
        timeout_s=_env_int("ASYNC_TIMEOUT_ONTOLOGY_SECONDS", 300),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        stale_grace_s=_env_int("ASYNC_STALE_GRACE_ONTOLOGY_SECONDS", 60),
        error_message="Ontology query failed",
    ),
    "short_screen": JobSpec(
        job_type="short_screen",
        request_model="api.routers.short_screen.ShortScreenRequest",
        compute_func="api.routers.short_screen._compute_short_screen",
        cache_key_func="api.routers.short_screen._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_SCREENS", "screens"),
        timeout_s=_env_int("ASYNC_TIMEOUT_SCREEN_SECONDS", 45 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Short screen failed",
        supports_progress=True,
        initial_progress={"phase": "queued", "done": 0, "total": 0},
    ),
    "long_screen": JobSpec(
        job_type="long_screen",
        request_model="api.routers.long_screen.LongScreenRequest",
        compute_func="api.routers.long_screen._compute_long_screen",
        cache_key_func="api.routers.long_screen._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_SCREENS", "screens"),
        timeout_s=_env_int("ASYNC_TIMEOUT_SCREEN_SECONDS", 45 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Long screen failed",
        supports_progress=True,
        initial_progress={"phase": "queued", "done": 0, "total": 0},
    ),
    "fundamental_momentum": JobSpec(
        job_type="fundamental_momentum",
        request_model="api.routers.fundamental_momentum.FMRequest",
        compute_func="api.routers.fundamental_momentum._compute_fundamental_momentum",
        cache_key_func="api.routers.fundamental_momentum._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_SCREENS", "screens"),
        timeout_s=_env_int("ASYNC_TIMEOUT_FUNDAMENTAL_MOMENTUM_SECONDS", 10 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Fundamental momentum failed",
    ),
    "price_momentum": JobSpec(
        job_type="price_momentum",
        request_model="api.routers.price_momentum.PriceMomentumRequest",
        compute_func="api.routers.price_momentum._compute_price_momentum",
        cache_key_func="api.routers.price_momentum._cache_key",
        queue_name=_env_queue("ASYNC_QUEUE_SCREENS", "screens"),
        timeout_s=_env_int("ASYNC_TIMEOUT_PRICE_MOMENTUM_SECONDS", 10 * 60),
        completed_ttl_s=DEFAULT_COMPLETED_TTL_S,
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Price momentum failed",
        supports_progress=True,
        initial_progress={"phase": "queued", "done": 0, "total": 0},
    ),
    "cache_warm": JobSpec(
        job_type="cache_warm",
        request_model=None,
        compute_func="api.maintenance_jobs.warm_caches",
        cache_key_func=None,
        queue_name=_env_queue("ASYNC_QUEUE_MAINTENANCE", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_CACHE_WARM_SECONDS", 5 * 60),
        completed_ttl_s=_env_int("ASYNC_MAINTENANCE_COMPLETED_TTL_SECONDS", 60 * 60),
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Cache warm failed",
    ),
    "market_snapshot_refresh": JobSpec(
        job_type="market_snapshot_refresh",
        request_model=None,
        compute_func="api.market_snapshots.refresh_market_snapshots",
        cache_key_func=None,
        queue_name=_env_queue("ASYNC_QUEUE_MAINTENANCE", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_MARKET_SNAPSHOT_SECONDS", 15 * 60),
        completed_ttl_s=_env_int("ASYNC_MAINTENANCE_COMPLETED_TTL_SECONDS", 60 * 60),
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Market snapshot refresh failed",
    ),
    "async_job_sweep": JobSpec(
        job_type="async_job_sweep",
        request_model=None,
        compute_func="api.maintenance_jobs.sweep_async_jobs",
        cache_key_func=None,
        queue_name=_env_queue("ASYNC_QUEUE_MAINTENANCE", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_SWEEP_SECONDS", 5 * 60),
        completed_ttl_s=_env_int("ASYNC_MAINTENANCE_COMPLETED_TTL_SECONDS", 60 * 60),
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Async job sweep failed",
    ),
    "governance_outbox_drain": JobSpec(
        job_type="governance_outbox_drain",
        request_model=None,
        compute_func="api.maintenance_jobs.drain_governance_outbox",
        cache_key_func=None,
        queue_name=_env_queue("ASYNC_QUEUE_MAINTENANCE", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_GOVERNANCE_OUTBOX_SECONDS", 5 * 60),
        completed_ttl_s=_env_int("ASYNC_MAINTENANCE_COMPLETED_TTL_SECONDS", 60 * 60),
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Governance outbox drain failed",
    ),
    "watch_trigger_monitor": JobSpec(
        job_type="watch_trigger_monitor",
        request_model=None,
        compute_func="api.watch_trigger_monitor.run_watch_trigger_monitor",
        cache_key_func=None,
        queue_name=_env_queue("ASYNC_QUEUE_MAINTENANCE", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_WATCH_TRIGGER_MONITOR_SECONDS", 10 * 60),
        completed_ttl_s=_env_int("ASYNC_MAINTENANCE_COMPLETED_TTL_SECONDS", 60 * 60),
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Watch trigger monitor failed",
    ),
    "continuous_optimizer": JobSpec(
        job_type="continuous_optimizer",
        request_model=None,
        compute_func="api.continuous_optimizer.run_continuous_optimizer",
        cache_key_func=None,
        queue_name=_env_queue("ASYNC_QUEUE_MAINTENANCE", "default"),
        timeout_s=_env_int("ASYNC_TIMEOUT_CONTINUOUS_OPTIMIZER_SECONDS", 20 * 60),
        completed_ttl_s=_env_int("ASYNC_MAINTENANCE_COMPLETED_TTL_SECONDS", 60 * 60),
        failed_ttl_s=DEFAULT_FAILED_TTL_S,
        error_message="Continuous optimizer failed",
    ),
}


def get_job_spec(job_type: str) -> JobSpec:
    try:
        return JOB_SPECS[job_type]
    except KeyError as exc:
        raise ValueError(f"Unknown async job type: {job_type}") from exc
