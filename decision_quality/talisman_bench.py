"""TalismanBench orchestrator for owned-model release gating."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.bench_openai_client import BenchOpenAIConfig, activate_bench_openai
from decision_quality.chat_eval_runner import (
    ChatEvalCase,
    run_agent_chat_in_process,
)
from decision_quality.chat_eval_runner import (
    build_report as build_chat_report,
)
from decision_quality.chat_eval_runner import (
    load_cases as load_chat_cases,
)
from decision_quality.chat_eval_runner import (
    run_case as run_chat_case,
)
from decision_quality.chat_eval_runner import (
    validate_case_input_refs as validate_chat_input_refs,
)
from decision_quality.eval_corpus import (
    build_baseline_report,
    compare_reports,
    load_baseline,
    validate_approved_case_metadata,
)
from decision_quality.eval_runner import (
    EvalCase as StructuredEvalCase,
)
from decision_quality.eval_runner import (
    build_report as build_structured_report,
)
from decision_quality.eval_runner import (
    load_cases as load_structured_cases,
)
from decision_quality.eval_runner import (
    run_case as run_structured_case,
)
from decision_quality.eval_runner import (
    validate_case_input_refs as validate_structured_input_refs,
)
from decision_quality.opportunity_candidate_eval_runner import (
    EvalCase as OpportunityEvalCase,
)
from decision_quality.opportunity_candidate_eval_runner import (
    build_report as build_opportunity_report,
)
from decision_quality.opportunity_candidate_eval_runner import (
    load_cases as load_opportunity_cases,
)
from decision_quality.opportunity_candidate_eval_runner import (
    run_case as run_opportunity_case,
)
from decision_quality.opportunity_candidate_eval_runner import (
    validate_case_input_refs as validate_opportunity_input_refs,
)
from decision_quality.supervised_labels import assign_split, check_split_leakage, split_group_for_case

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "docs" / "talisman_bench" / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "talisman_bench"

REQUIRED_MANIFEST_KEYS = (
    "manifest_version",
    "benchmark_version",
    "name",
    "corpora",
    "split_policy",
    "hard_blockers",
    "scored_metrics",
    "graduation_thresholds",
    "baseline_regression",
    "baseline_model",
    "candidate_model",
)
REQUIRED_CORPUS_KEYS = (
    "id",
    "runner",
    "cases_dir",
    "baseline_path",
    "dimensions",
    "fail_under_deterministic",
)
VALID_RUNNERS = {"structured", "chat", "opportunity_candidate"}


@dataclass(frozen=True)
class CorpusConfig:
    id: str
    runner: str
    cases_dir: Path
    baseline_path: Path
    dimensions: tuple[str, ...]
    fail_under_deterministic: float
    fail_under_judge: float | None
    judge_default: bool


@dataclass(frozen=True)
class ModelTarget:
    label: str
    provider: str | None = None
    tier: str | None = None
    candidate_config: BenchOpenAIConfig | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _resolve_path(value: str, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _parse_corpus_config(entry: dict[str, Any], *, root: Path = ROOT) -> CorpusConfig:
    missing = [key for key in REQUIRED_CORPUS_KEYS if key not in entry]
    if missing:
        raise ValueError(f"corpus {entry.get('id', '<unknown>')} missing keys: {', '.join(missing)}")
    runner = str(entry["runner"])
    if runner not in VALID_RUNNERS:
        raise ValueError(f"unsupported corpus runner: {runner}")
    return CorpusConfig(
        id=str(entry["id"]),
        runner=runner,
        cases_dir=_resolve_path(str(entry["cases_dir"]), root=root),
        baseline_path=_resolve_path(str(entry["baseline_path"]), root=root),
        dimensions=tuple(str(item) for item in entry.get("dimensions") or []),
        fail_under_deterministic=float(entry["fail_under_deterministic"]),
        fail_under_judge=float(entry["fail_under_judge"]) if entry.get("fail_under_judge") is not None else None,
        judge_default=bool(entry.get("judge_default", False)),
    )


def validate_manifest(manifest: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            errors.append(f"manifest missing required key: {key}")

    corpora = manifest.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        errors.append("manifest.corpora must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    for entry in corpora:
        if not isinstance(entry, dict):
            errors.append("manifest.corpora entries must be objects")
            continue
        try:
            config = _parse_corpus_config(entry, root=root)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if config.id in seen_ids:
            errors.append(f"duplicate corpus id: {config.id}")
        seen_ids.add(config.id)
        if not config.cases_dir.exists():
            errors.append(f"corpus {config.id} cases_dir does not exist: {config.cases_dir}")
        if not config.baseline_path.exists():
            errors.append(f"corpus {config.id} baseline_path does not exist: {config.baseline_path}")

    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, dict):
        errors.append("manifest.split_policy must be an object")
    else:
        for key in ("train_ratio", "val_ratio", "training_splits", "held_out_splits"):
            if key not in split_policy:
                errors.append(f"manifest.split_policy missing {key}")

    return errors


def corpus_configs(manifest: dict[str, Any], *, root: Path = ROOT) -> list[CorpusConfig]:
    corpora = manifest.get("corpora")
    if not isinstance(corpora, list):
        raise ValueError("manifest.corpora must be a list")
    return [_parse_corpus_config(entry, root=root) for entry in corpora if isinstance(entry, dict)]


def _load_cases_for_corpus(config: CorpusConfig, *, statuses: set[str]) -> list[Any]:
    if config.runner == "structured":
        return load_structured_cases(statuses=statuses, cases_dir=config.cases_dir)
    if config.runner == "chat":
        return load_chat_cases(statuses=statuses, cases_dir=config.cases_dir)
    return load_opportunity_cases(statuses=statuses, cases_dir=config.cases_dir)


def _validate_input_refs_for_case(config: CorpusConfig, case: Any) -> list[str]:
    if config.runner == "structured":
        return validate_structured_input_refs(case)
    if config.runner == "chat":
        return validate_chat_input_refs(case)
    return validate_opportunity_input_refs(case)


def build_inventory(
    manifest: dict[str, Any],
    *,
    approved_only: bool,
    root: Path = ROOT,
) -> dict[str, Any]:
    split_policy = manifest.get("split_policy") or {}
    train_ratio = float(split_policy.get("train_ratio", 0.7))
    val_ratio = float(split_policy.get("val_ratio", 0.15))
    statuses = {"approved"} if approved_only else {"review", "approved"}

    rows: list[dict[str, Any]] = []
    corpora_summary: dict[str, Any] = {}
    for config in corpus_configs(manifest, root=root):
        cases = _load_cases_for_corpus(config, statuses=statuses)
        corpus_rows: list[dict[str, Any]] = []
        for case in cases:
            case_id = case.case_id
            case_data = case.data
            split_group = split_group_for_case(case_id=case_id, case_data=case_data)
            split = assign_split(split_group, train_ratio=train_ratio, val_ratio=val_ratio)
            row = {
                "corpus_id": config.id,
                "runner": config.runner,
                "case_id": case_id,
                "case_path": str(case.path.relative_to(root) if case.path.is_relative_to(root) else case.path),
                "status": case.status,
                "split_group": split_group,
                "split": split,
                "corpus_tags": list(case_data.get("corpus_tags") or []),
                "failure_type": case_data.get("failure_type"),
            }
            rows.append(row)
            corpus_rows.append(row)
        corpora_summary[config.id] = {
            "runner": config.runner,
            "case_count": len(corpus_rows),
            "held_out_case_count": sum(1 for row in corpus_rows if row["split"] == "holdout"),
        }

    leakage_violations = check_split_leakage(rows)
    return {
        "generated_at": _now_iso(),
        "approved_only": approved_only,
        "case_count": len(rows),
        "corpora": corpora_summary,
        "rows": rows,
        "leakage_violations": leakage_violations,
    }


def run_structural_gates(
    manifest: dict[str, Any],
    *,
    approved_only: bool,
    root: Path = ROOT,
) -> dict[str, Any]:
    statuses = {"approved"} if approved_only else {"review", "approved"}
    errors: list[str] = []
    corpora_reports: dict[str, Any] = {}

    for config in corpus_configs(manifest, root=root):
        cases = _load_cases_for_corpus(config, statuses=statuses)
        corpus_errors: list[str] = []
        for case in cases:
            metadata_errors = validate_approved_case_metadata(case.data) if case.status == "approved" else []
            for message in metadata_errors:
                corpus_errors.append(f"{case.case_id}: {message}")
            ref_errors = _validate_input_refs_for_case(config, case)
            for message in ref_errors:
                corpus_errors.append(f"{case.case_id}: {message}")

        if approved_only:
            try:
                baseline = load_baseline(config.baseline_path)
            except Exception as exc:
                corpus_errors.append(f"baseline load failed: {exc}")
            else:
                approved_ids = {case.case_id for case in cases}
                baseline_ids = set((baseline.get("cases") or {}).keys())
                missing = sorted(approved_ids - baseline_ids)
                extra = sorted(baseline_ids - approved_ids)
                if missing:
                    corpus_errors.append(f"missing baseline inventory for approved cases: {', '.join(missing)}")
                if extra:
                    corpus_errors.append(f"baseline contains non-approved cases: {', '.join(extra)}")

        corpora_reports[config.id] = {
            "case_count": len(cases),
            "errors": corpus_errors,
            "passed": not corpus_errors,
        }
        errors.extend(f"{config.id}: {message}" for message in corpus_errors)

    return {
        "generated_at": _now_iso(),
        "passed": not errors,
        "errors": errors,
        "corpora": corpora_reports,
    }


def _run_structured_cases(
    cases: list[StructuredEvalCase],
    *,
    model_target: ModelTarget,
    dry_run: bool,
    judge: bool,
    fail_under_judge: float,
) -> list[dict[str, Any]]:
    model = model_target.tier or "high"
    provider = model_target.provider
    return [
        run_structured_case(
            case,
            model=model,
            provider=provider,
            judge=judge,
            dry_run=dry_run,
            fail_under_judge=fail_under_judge,
        )
        for case in cases
    ]


def _run_chat_cases(
    cases: list[ChatEvalCase],
    *,
    model_target: ModelTarget,
    dry_run: bool,
    judge: bool,
) -> list[dict[str, Any]]:
    model = model_target.tier or "high"
    provider = model_target.provider

    def runner(case: ChatEvalCase) -> Any:
        return run_agent_chat_in_process(case)

    return [
        run_chat_case(
            case,
            agent_runner=runner,
            judge=judge,
            model=model,
            provider=provider,
            dry_run=dry_run,
        )
        for case in cases
    ]


def _run_opportunity_cases(
    cases: list[OpportunityEvalCase],
    *,
    model_target: ModelTarget,
    dry_run: bool,
    judge: bool,
    fail_under_judge: float,
) -> list[dict[str, Any]]:
    model = model_target.tier or "high"
    provider = model_target.provider
    return [
        run_opportunity_case(
            case,
            model=model,
            provider=provider,
            judge=judge,
            dry_run=dry_run,
            fail_under_judge=fail_under_judge,
        )
        for case in cases
    ]


def run_corpus(
    config: CorpusConfig,
    *,
    statuses: set[str],
    model_target: ModelTarget,
    dry_run: bool,
    judge_override: bool | None = None,
) -> dict[str, Any]:
    cases = _load_cases_for_corpus(config, statuses=statuses)
    judge = config.judge_default if judge_override is None else judge_override
    fail_under_judge = config.fail_under_judge if config.fail_under_judge is not None else 14.0

    def execute() -> list[dict[str, Any]]:
        if config.runner == "structured":
            return _run_structured_cases(
                cases,
                model_target=model_target,
                dry_run=dry_run,
                judge=judge,
                fail_under_judge=fail_under_judge,
            )
        if config.runner == "chat":
            return _run_chat_cases(cases, model_target=model_target, dry_run=dry_run, judge=judge)
        return _run_opportunity_cases(
            cases,
            model_target=model_target,
            dry_run=dry_run,
            judge=judge,
            fail_under_judge=fail_under_judge,
        )

    if model_target.candidate_config is not None:
        with activate_bench_openai(model_target.candidate_config):
            results = execute()
    else:
        results = execute()

    if config.runner == "chat":
        report = build_chat_report(results, fail_under_deterministic=config.fail_under_deterministic)
    elif config.runner == "structured":
        report = build_structured_report(
            results,
            fail_under_deterministic=config.fail_under_deterministic,
            fail_under_judge=fail_under_judge,
        )
    else:
        report = build_opportunity_report(
            results,
            fail_under_deterministic=config.fail_under_deterministic,
            fail_under_judge=fail_under_judge,
        )

    baseline_summary = build_baseline_report(
        results,
        status_filter=statuses,
        notes=f"TalismanBench corpus {config.id} run.",
    )
    return {
        "corpus_id": config.id,
        "runner": config.runner,
        "model_target": model_target.label,
        "dry_run": dry_run,
        "case_count": len(results),
        "report": report,
        "baseline_summary": baseline_summary,
    }


def _deterministic_pass_rate(report: dict[str, Any]) -> float:
    cases = report.get("cases")
    if isinstance(cases, list):
        entries = cases
    elif isinstance(cases, dict):
        entries = list(cases.values())
    else:
        return 0.0
    if not entries:
        return 0.0
    passed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        deterministic = entry.get("deterministic") or {}
        if entry.get("dry_run"):
            passed += 1
            continue
        if deterministic.get("passed") is True:
            passed += 1
        elif entry.get("deterministic_passed") is True:
            passed += 1
    return passed / len(entries)


def _judge_total_mean(report: dict[str, Any]) -> float | None:
    totals: list[float] = []
    cases = report.get("cases")
    entries = cases if isinstance(cases, list) else list((cases or {}).values()) if isinstance(cases, dict) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        judge = entry.get("judge") or {}
        total = judge.get("total")
        if total is None:
            total = entry.get("judge_total")
        if isinstance(total, (int, float)):
            totals.append(float(total))
    if not totals:
        return None
    return statistics.fmean(totals)


def _latency_p95_ms(corpus_runs: list[dict[str, Any]]) -> float | None:
    latencies: list[float] = []
    for run in corpus_runs:
        report = run.get("report") or {}
        cases = report.get("cases")
        entries = cases if isinstance(cases, list) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            elapsed = entry.get("elapsed_ms")
            if isinstance(elapsed, (int, float)):
                latencies.append(float(elapsed))
    if not latencies:
        return None
    latencies.sort()
    index = max(0, int(round(0.95 * (len(latencies) - 1))))
    return latencies[index]


def _aggregate_scored_metrics(corpus_runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = sum(int(run.get("case_count") or 0) for run in corpus_runs)
    weighted_pass = 0.0
    judge_values: list[float] = []
    for run in corpus_runs:
        report = run.get("report") or {}
        case_count = int(run.get("case_count") or 0)
        weighted_pass += _deterministic_pass_rate(report) * case_count
        judge_mean = _judge_total_mean(report)
        if judge_mean is not None:
            judge_values.append(judge_mean)
    return {
        "deterministic_pass_rate": (weighted_pass / total_cases) if total_cases else 0.0,
        "judge_total_mean": statistics.fmean(judge_values) if judge_values else None,
        "latency_p95_ms": _latency_p95_ms(corpus_runs),
        "token_use_total": None,
        "estimated_cost_usd": None,
    }


def _hard_blocker_results(
    *,
    manifest: dict[str, Any],
    manifest_errors: list[str],
    inventory: dict[str, Any],
    structural: dict[str, Any],
    baseline_runs: list[dict[str, Any]] | None,
    candidate_runs: list[dict[str, Any]] | None,
    comparisons: dict[str, Any] | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    blockers = [str(item) for item in manifest.get("hard_blockers", []) if isinstance(item, str)]
    results: list[dict[str, Any]] = []

    def add(blocker_id: str, passed: bool, message: str = "") -> None:
        if blocker_id not in blockers:
            return
        results.append({"id": blocker_id, "passed": passed, "message": message})

    add("manifest_invalid", not manifest_errors, "; ".join(manifest_errors))
    leakage = inventory.get("leakage_violations") or []
    add(
        "inventory_leakage",
        not leakage,
        f"split-group leakage detected: {', '.join(leakage)}" if leakage else "",
    )
    add(
        "structural_gate_failure",
        bool(structural.get("passed")),
        "; ".join(structural.get("errors") or []),
    )

    missing_inventory_errors: list[str] = []
    for corpus_id, payload in (structural.get("corpora") or {}).items():
        for message in payload.get("errors") or []:
            if "baseline inventory" in message or "baseline contains" in message:
                missing_inventory_errors.append(f"{corpus_id}: {message}")
    add(
        "missing_baseline_inventory",
        not missing_inventory_errors,
        "; ".join(missing_inventory_errors),
    )

    if not dry_run and candidate_runs is not None:
        deterministic_failures: list[str] = []
        for run in candidate_runs:
            summary = (run.get("report") or {}).get("summary") or {}
            for case_id in summary.get("deterministic_failures") or []:
                deterministic_failures.append(f"{run['corpus_id']}:{case_id}")
        add(
            "deterministic_failure",
            not deterministic_failures,
            ", ".join(deterministic_failures),
        )

    if comparisons is not None:
        regression_cases: list[str] = []
        for _corpus_id, comparison in comparisons.items():
            summary = comparison.get("summary") or {}
            if summary.get("regression_detected"):
                regression_cases.extend(summary.get("new_deterministic_failures") or [])
        add(
            "baseline_regression",
            not regression_cases,
            ", ".join(regression_cases),
        )
    elif baseline_runs is not None and not dry_run:
        add("baseline_regression", True, "")

    return results


def build_release_report(
    *,
    manifest: dict[str, Any],
    manifest_errors: list[str],
    inventory: dict[str, Any],
    structural: dict[str, Any],
    dry_run: bool,
    baseline_runs: list[dict[str, Any]] | None = None,
    candidate_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    comparisons: dict[str, Any] | None = None
    if baseline_runs is not None and candidate_runs is not None and not dry_run:
        comparisons = {}
        candidate_by_id = {run["corpus_id"]: run for run in candidate_runs}
        for baseline_run in baseline_runs:
            corpus_id = baseline_run["corpus_id"]
            candidate_run = candidate_by_id.get(corpus_id)
            if candidate_run is None:
                continue
            config = next(item for item in corpus_configs(manifest) if item.id == corpus_id)
            committed_baseline = load_baseline(config.baseline_path)
            comparisons[corpus_id] = compare_reports(
                committed_baseline,
                candidate_run["baseline_summary"],
            )

    hard_blockers = _hard_blocker_results(
        manifest=manifest,
        manifest_errors=manifest_errors,
        inventory=inventory,
        structural=structural,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        comparisons=comparisons,
        dry_run=dry_run,
    )
    scored_metrics = {
        "baseline": _aggregate_scored_metrics(baseline_runs or []),
        "candidate": _aggregate_scored_metrics(candidate_runs or []),
    }
    if baseline_runs and candidate_runs:
        base_rate = scored_metrics["baseline"]["deterministic_pass_rate"]
        cand_rate = scored_metrics["candidate"]["deterministic_pass_rate"]
        scored_metrics["delta"] = {
            "deterministic_pass_rate": cand_rate - base_rate,
        }

    thresholds = manifest.get("graduation_thresholds") or {}
    threshold_failures: list[str] = []
    candidate_rate = scored_metrics["candidate"]["deterministic_pass_rate"]
    if not dry_run and candidate_runs:
        min_rate = float(thresholds.get("min_deterministic_pass_rate", 0.95))
        if candidate_rate < min_rate:
            threshold_failures.append(f"deterministic_pass_rate {candidate_rate:.3f} below minimum {min_rate:.3f}")
        max_new_failures = int(thresholds.get("max_new_deterministic_failures", 0))
        if comparisons:
            new_failures = sum(
                len((comparison.get("summary") or {}).get("new_deterministic_failures") or [])
                for comparison in comparisons.values()
            )
            if new_failures > max_new_failures:
                threshold_failures.append(
                    f"new deterministic failures {new_failures} exceed allowed {max_new_failures}"
                )

    hard_blocker_failures = [item for item in hard_blockers if not item.get("passed")]
    release_passed = not hard_blocker_failures and not threshold_failures

    return {
        "manifest_version": manifest.get("manifest_version"),
        "benchmark_version": manifest.get("benchmark_version"),
        "benchmark_name": manifest.get("name"),
        "generated_at": _now_iso(),
        "mode": "dry_run" if dry_run else "release",
        "inventory": {
            "case_count": inventory.get("case_count"),
            "corpora": inventory.get("corpora"),
            "leakage_violations": inventory.get("leakage_violations"),
        },
        "structural_gates": structural,
        "hard_blockers": hard_blockers,
        "scored_metrics": scored_metrics,
        "baseline": {"corpora": baseline_runs or []},
        "candidate": {"corpora": candidate_runs or []},
        "comparison": comparisons,
        "release_gate": {
            "passed": release_passed,
            "hard_blocker_failures": [item["id"] for item in hard_blocker_failures],
            "threshold_failures": threshold_failures,
        },
    }


def _default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"talisman_bench_{timestamp}"


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def _baseline_model_target(manifest: dict[str, Any], *, tier_override: str | None = None) -> ModelTarget:
    baseline_model = manifest.get("baseline_model") or {}
    provider = str(baseline_model.get("provider") or "openai")
    tier = tier_override or str(baseline_model.get("tier") or "mid")
    return ModelTarget(label="baseline", provider=provider, tier=tier)


def _candidate_model_target(
    manifest: dict[str, Any],
    *,
    base_url: str | None,
    api_key_env: str | None,
    model: str | None,
) -> ModelTarget:
    candidate_model = manifest.get("candidate_model") or {}
    config = BenchOpenAIConfig.from_env(
        base_url=base_url,
        api_key_env=api_key_env or str(candidate_model.get("api_key_env") or "TALISMAN_BENCH_CANDIDATE_API_KEY"),
        model=model,
        model_env=str(candidate_model.get("model_env") or "TALISMAN_BENCH_CANDIDATE_MODEL"),
    )
    return ModelTarget(label="candidate", candidate_config=config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TalismanBench release gates for owned agent models.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to TalismanBench manifest JSON.")
    parser.add_argument("--approved-only", action="store_true", help="Include only approved corpus cases.")
    parser.add_argument("--dry-run", action="store_true", help="Run offline structural and leakage gates only.")
    parser.add_argument("--output", default=None, help="Output directory for release report artifacts.")
    parser.add_argument("--baseline-model", default=None, help="Baseline model tier override (low|mid|high).")
    parser.add_argument("--baseline-provider", default=None, help="Baseline provider override.")
    parser.add_argument("--candidate-openai-base-url", default=None, help="OpenAI-compatible candidate base URL.")
    parser.add_argument(
        "--candidate-api-key-env",
        default=None,
        help="Environment variable containing the candidate API key.",
    )
    parser.add_argument(
        "--candidate-model", default=None, help="Candidate model id for the OpenAI-compatible endpoint."
    )
    parser.add_argument("--judge", dest="judge", action="store_true", default=False)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _load_local_env()
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    manifest_errors = validate_manifest(manifest, root=ROOT)
    statuses = {"approved"} if args.approved_only else {"review", "approved"}

    inventory = build_inventory(manifest, approved_only=args.approved_only, root=ROOT)
    structural = run_structural_gates(manifest, approved_only=args.approved_only, root=ROOT)

    baseline_runs: list[dict[str, Any]] | None = None
    candidate_runs: list[dict[str, Any]] | None = None

    if args.dry_run:
        for config in corpus_configs(manifest, root=ROOT):
            target = ModelTarget(label="dry_run")
            run_payload = run_corpus(
                config,
                statuses=statuses,
                model_target=target,
                dry_run=True,
                judge_override=False,
            )
            if baseline_runs is None:
                baseline_runs = []
            baseline_runs.append(run_payload)
    else:
        baseline_target = _baseline_model_target(manifest, tier_override=args.baseline_model)
        if args.baseline_provider:
            baseline_target = ModelTarget(
                label="baseline",
                provider=args.baseline_provider,
                tier=baseline_target.tier,
            )
        candidate_target = _candidate_model_target(
            manifest,
            base_url=args.candidate_openai_base_url,
            api_key_env=args.candidate_api_key_env,
            model=args.candidate_model,
        )
        baseline_runs = []
        candidate_runs = []
        for config in corpus_configs(manifest, root=ROOT):
            baseline_runs.append(
                run_corpus(
                    config,
                    statuses=statuses,
                    model_target=baseline_target,
                    dry_run=False,
                    judge_override=args.judge,
                )
            )
            candidate_runs.append(
                run_corpus(
                    config,
                    statuses=statuses,
                    model_target=candidate_target,
                    dry_run=False,
                    judge_override=args.judge,
                )
            )

    release_report = build_release_report(
        manifest=manifest,
        manifest_errors=manifest_errors,
        inventory=inventory,
        structural=structural,
        dry_run=args.dry_run,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
    )

    output_dir = Path(args.output) if args.output else _default_output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "release_report.json"
    inventory_path = output_dir / "inventory.json"
    report_path.write_text(json.dumps(release_report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote TalismanBench release report: {report_path}")

    if manifest_errors:
        print("Manifest validation errors:")
        for message in manifest_errors:
            print(f"  - {message}")
    if not structural.get("passed"):
        print("Structural gate failures detected.")
    if inventory.get("leakage_violations"):
        print("Split-group leakage detected.")

    return 0 if release_report["release_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
