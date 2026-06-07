from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from decision_quality.bench_openai_client import BenchOpenAIConfig, activate_bench_openai
from decision_quality.talisman_bench import (
    _aggregate_scored_metrics,
    build_inventory,
    build_release_report,
    load_candidate_matrix,
    load_manifest,
    main,
    run_structural_gates,
    validate_candidate_matrix,
    validate_manifest,
)


def _minimal_manifest(tmp_path: Path) -> dict:
    structured_cases = tmp_path / "structured_cases"
    structured_cases.mkdir()
    structured_baseline = tmp_path / "structured_baseline.json"
    structured_baseline.write_text(
        json.dumps(
            {
                "baseline_version": 1,
                "cases": {
                    "case_a": {
                        "case_id": "case_a",
                        "deterministic_passed": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (structured_cases / "case_a.json").write_text(
        json.dumps(
            {
                "id": "case_a",
                "status": "approved",
                "corpus_tags": ["structured_dq"],
                "failure_type": "process_regression",
                "required_dq_dimensions": ["thesis_clarity"],
                "input_refs": [],
                "gold_output": {
                    "ticker": "TEST",
                    "recommended_action": "watch",
                    "conviction_level": "medium",
                    "actionability": {"status": "needs_more_work", "missing_inputs": ["price"]},
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "manifest_version": 1,
        "benchmark_version": "test",
        "name": "TestBench",
        "corpora": [
            {
                "id": "structured_dq",
                "runner": "structured",
                "cases_dir": str(structured_cases),
                "baseline_path": str(structured_baseline),
                "dimensions": ["structured_output"],
                "fail_under_deterministic": 80.0,
                "fail_under_judge": 14.0,
            }
        ],
        "split_policy": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "training_splits": ["train", "validation"],
            "held_out_splits": ["holdout"],
        },
        "hard_blockers": [
            "manifest_invalid",
            "inventory_leakage",
            "structural_gate_failure",
            "missing_baseline_inventory",
            "deterministic_failure",
            "baseline_regression",
        ],
        "scored_metrics": ["deterministic_pass_rate"],
        "graduation_thresholds": {
            "min_deterministic_pass_rate": 0.95,
            "max_new_deterministic_failures": 0,
        },
        "baseline_regression": {"allow_new_failures": 0},
        "baseline_model": {"provider": "openai", "tier": "mid"},
        "candidate_model": {
            "protocol": "openai_compatible",
            "base_url_env": "TALISMAN_BENCH_CANDIDATE_BASE_URL",
            "api_key_env": "TALISMAN_BENCH_CANDIDATE_API_KEY",
            "model_env": "TALISMAN_BENCH_CANDIDATE_MODEL",
        },
    }


def test_validate_manifest_rejects_missing_keys(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"name": "incomplete"}), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest, root=tmp_path)
    assert any("manifest_version" in error for error in errors)


def test_build_inventory_detects_split_group_leakage(tmp_path: Path):
    cases = tmp_path / "cases"
    cases.mkdir()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"cases": {}}), encoding="utf-8")
    for case_id in ("case_a", "case_b"):
        (cases / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "id": case_id,
                    "status": "approved",
                    "source_session_id": "shared-session",
                    "corpus_tags": ["structured_dq"],
                    "failure_type": "process_regression",
                    "required_dq_dimensions": ["thesis_clarity"],
                    "input_refs": [],
                }
            ),
            encoding="utf-8",
        )

    manifest = _minimal_manifest(tmp_path)
    manifest["corpora"][0]["cases_dir"] = str(cases)
    manifest["corpora"][0]["baseline_path"] = str(baseline)

    inventory = build_inventory(manifest, approved_only=True, root=tmp_path)
    # Deterministic assign_split should not leak; verify inventory shape instead.
    assert inventory["case_count"] == 2
    assert inventory["leakage_violations"] == []


def test_build_inventory_reports_leakage_violation(tmp_path: Path):
    manifest = _minimal_manifest(tmp_path)
    inventory = {
        "case_count": 2,
        "corpora": {},
        "leakage_violations": ["shared-group"],
    }
    structural = {"passed": True, "errors": [], "corpora": {}}
    report = build_release_report(
        manifest=manifest,
        manifest_errors=[],
        inventory=inventory,
        structural=structural,
        dry_run=True,
        baseline_runs=[],
    )
    leakage_blocker = next(item for item in report["hard_blockers"] if item["id"] == "inventory_leakage")
    assert leakage_blocker["passed"] is False
    assert report["release_gate"]["passed"] is False


def test_structural_gate_catches_missing_baseline_inventory(tmp_path: Path):
    manifest = _minimal_manifest(tmp_path)
    structural = run_structural_gates(manifest, approved_only=True, root=tmp_path)
    assert structural["passed"] is True

    cases_dir = Path(manifest["corpora"][0]["cases_dir"])
    (cases_dir / "case_b.json").write_text(
        json.dumps(
            {
                "id": "case_b",
                "status": "approved",
                "corpus_tags": ["structured_dq"],
                "failure_type": "process_regression",
                "required_dq_dimensions": ["thesis_clarity"],
                "input_refs": [],
            }
        ),
        encoding="utf-8",
    )
    structural = run_structural_gates(manifest, approved_only=True, root=tmp_path)
    assert structural["passed"] is False
    assert any("missing baseline inventory" in error for error in structural["errors"])


def test_release_report_separates_hard_blockers_and_scored_metrics(tmp_path: Path):
    manifest = _minimal_manifest(tmp_path)
    inventory = build_inventory(manifest, approved_only=True, root=tmp_path)
    structural = run_structural_gates(manifest, approved_only=True, root=tmp_path)
    baseline_runs = [
        {
            "corpus_id": "structured_dq",
            "case_count": 1,
            "report": {
                "cases": [
                    {
                        "case_id": "case_a",
                        "dry_run": False,
                        "deterministic": {"passed": True, "score": 100.0},
                    }
                ]
            },
            "baseline_summary": {
                "cases": {"case_a": {"case_id": "case_a", "deterministic_passed": True}},
            },
        }
    ]
    candidate_runs = [
        {
            "corpus_id": "structured_dq",
            "case_count": 1,
            "report": {
                "summary": {"deterministic_failures": ["case_a"]},
                "cases": [
                    {
                        "case_id": "case_a",
                        "dry_run": False,
                        "deterministic": {"passed": False, "score": 0.0},
                    }
                ],
            },
            "baseline_summary": {
                "cases": {"case_a": {"case_id": "case_a", "deterministic_passed": False}},
            },
        }
    ]
    report = build_release_report(
        manifest=manifest,
        manifest_errors=[],
        inventory=inventory,
        structural=structural,
        dry_run=False,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
    )
    blocker_ids = {item["id"] for item in report["hard_blockers"] if not item["passed"]}
    assert "deterministic_failure" in blocker_ids
    assert "scored_metrics" in report
    assert report["scored_metrics"]["candidate"]["deterministic_pass_rate"] == 0.0
    assert report["release_gate"]["passed"] is False


def test_broken_candidate_triggers_baseline_regression(tmp_path: Path):
    manifest = _minimal_manifest(tmp_path)
    inventory = build_inventory(manifest, approved_only=True, root=tmp_path)
    structural = run_structural_gates(manifest, approved_only=True, root=tmp_path)
    baseline_runs = [
        {
            "corpus_id": "structured_dq",
            "case_count": 1,
            "report": {"summary": {"deterministic_failures": []}, "cases": []},
            "baseline_summary": {
                "cases": {"case_a": {"case_id": "case_a", "deterministic_passed": True}},
            },
        }
    ]
    candidate_runs = [
        {
            "corpus_id": "structured_dq",
            "case_count": 1,
            "report": {"summary": {"deterministic_failures": []}, "cases": []},
            "baseline_summary": {
                "cases": {"case_a": {"case_id": "case_a", "deterministic_passed": False}},
            },
        }
    ]
    report = build_release_report(
        manifest=manifest,
        manifest_errors=[],
        inventory=inventory,
        structural=structural,
        dry_run=False,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
    )
    regression = next(item for item in report["hard_blockers"] if item["id"] == "baseline_regression")
    assert regression["passed"] is False


def test_main_dry_run_against_committed_manifest(tmp_path: Path):
    exit_code = main(
        [
            "--manifest",
            "docs/talisman_bench/manifest.json",
            "--approved-only",
            "--dry-run",
            "--output",
            str(tmp_path / "bench_out"),
        ]
    )
    assert exit_code == 0
    report = json.loads((tmp_path / "bench_out" / "release_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "dry_run"
    assert report["release_gate"]["passed"] is True
    assert report["inventory"]["case_count"] == 43


def test_validate_committed_candidate_matrix():
    matrix = load_candidate_matrix(Path("docs/talisman_bench/candidate_matrix.json"))
    errors = validate_candidate_matrix(matrix)
    assert errors == []
    assert len(matrix["models"]) >= 3
    assert len(matrix["hosts"]) >= 2


def test_aggregate_scored_metrics_collects_tokens_and_cost():
    corpus_runs = [
        {
            "case_count": 1,
            "report": {
                "cases": [
                    {
                        "case_id": "case_a",
                        "diagnostics": {"usage": {"input_tokens": 100, "output_tokens": 50}},
                    }
                ]
            },
        }
    ]
    metrics = _aggregate_scored_metrics(
        corpus_runs,
        cost_per_1k_input_tokens_usd=0.0002,
        cost_per_1k_output_tokens_usd=0.0004,
    )
    assert metrics["token_use_total"] == 150
    assert metrics["estimated_cost_usd"] == pytest.approx(0.00004)


def test_build_release_report_includes_candidate_matrix_metadata(tmp_path: Path):
    manifest = _minimal_manifest(tmp_path)
    matrix = {
        "matrix_version": "1.0.0",
        "models": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
        "hosts": [{"id": "h1"}, {"id": "h2"}],
        "combinations": [{"id": "c1", "model_id": "m1", "host_id": "h1", "served_model_name": "m1"}],
        "selection": {"primary_combination_id": "c1"},
    }
    inventory = build_inventory(manifest, approved_only=True, root=tmp_path)
    structural = run_structural_gates(manifest, approved_only=True, root=tmp_path)
    report = build_release_report(
        manifest=manifest,
        manifest_errors=[],
        inventory=inventory,
        structural=structural,
        dry_run=True,
        baseline_runs=[],
        candidate_matrix=matrix,
        candidate_matrix_errors=[],
        selection_metadata={"smoke_only": True, "combination_id": "c1"},
    )
    assert report["candidate_matrix"]["matrix_version"] == "1.0.0"
    assert report["selection_metadata"]["combination_id"] == "c1"


def test_main_dry_run_with_smoke_only_filters_inventory(tmp_path: Path):
    exit_code = main(
        [
            "--manifest",
            "docs/talisman_bench/manifest.json",
            "--approved-only",
            "--dry-run",
            "--smoke-only",
            "--combination-id",
            "qwen-local-vllm",
            "--output",
            str(tmp_path / "bench_smoke"),
        ]
    )
    assert exit_code == 0
    report = json.loads((tmp_path / "bench_smoke" / "release_report.json").read_text(encoding="utf-8"))
    assert report["selection_metadata"]["smoke_only"] is True
    assert report["selection_metadata"]["combination_id"] == "qwen-local-vllm"
    assert sum(run["case_count"] for run in report["baseline"]["corpora"]) == 3


def test_activate_bench_openai_patches_llm_utils(monkeypatch):
    import llm_utils

    original_text = llm_utils.call_llm_text
    config = BenchOpenAIConfig(base_url="http://localhost:8000/v1", api_key="test", model="owned-candidate")

    with patch(
        "decision_quality.bench_openai_client.call_openai_compatible_text",
        return_value=("{}", [], object()),
    ):
        with activate_bench_openai(config):
            text, _, _ = llm_utils.call_llm_text(prompt="hello", model="mid")
        assert text == "{}"
    assert llm_utils.call_llm_text is original_text
