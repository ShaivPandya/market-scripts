"""Export, train, and evaluate supervised DQ synthesis / opportunity triage models."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.chat_eval_runner import CASES_DIR as CHAT_CASES_DIR
from decision_quality.chat_eval_runner import ChatEvalCase
from decision_quality.chat_eval_runner import load_cases as load_chat_cases
from decision_quality.eval_corpus import TRAINING_EXPORT_STATUSES
from decision_quality.eval_runner import CASES_DIR as STRUCTURED_CASES_DIR
from decision_quality.eval_runner import EvalCase
from decision_quality.eval_runner import load_cases as load_structured_cases
from decision_quality.supervised_labels import (
    assign_split,
    build_row_provenance,
    check_split_leakage,
    extract_labels_from_row,
    labels_from_chat_eval,
    labels_from_opportunity_candidate_gold,
    labels_from_structured_dq_gold,
    row_is_training_eligible,
    split_group_for_case,
)
from decision_quality.synthesis_supervised import featurize_context_row, write_model_card

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "synthesis_supervised_training"
DEFAULT_MODEL_DIR = ROOT / "data" / "synthesis_supervised_models"
DEFAULT_REGISTRY_PATH = DEFAULT_MODEL_DIR / "registry.json"
OC_CASES_DIR = ROOT / "docs" / "opportunity_candidate_evals" / "cases"

DEFAULT_ROLLOUT_GATES = {
    "min_triage_accuracy": 0.70,
    "min_stance_accuracy": 0.65,
    "max_gate_failure_regression": 0,
}


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_oc_cases(*, statuses: set[str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not OC_CASES_DIR.exists():
        return cases
    for path in sorted(OC_CASES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(data.get("status") or "draft") not in statuses:
            continue
        cases.append({"case_id": str(data.get("id") or path.stem), "path": path, "data": data})
    return cases


def _context_features_from_case(case_data: dict[str, Any]) -> dict[str, Any]:
    gold = case_data.get("gold_output") if isinstance(case_data.get("gold_output"), dict) else {}
    context_pack = case_data.get("context_pack_expectations")
    pack_id = None
    pack_complete = True
    missing_count = 0
    if isinstance(context_pack, dict):
        pack_id = context_pack.get("expected_context_pack")
        if context_pack.get("expect_complete") is False:
            pack_complete = False
        missing_count = len(context_pack.get("required_missing_input_terms") or [])
    if isinstance(gold, dict):
        missing_count = max(missing_count, len(gold.get("missing_inputs") or []))
    return {
        "opportunity_type": gold.get("opportunity_type") if isinstance(gold, dict) else None,
        "context_pack_id": pack_id or case_data.get("tool_pack"),
        "context_pack_complete": pack_complete,
        "missing_input_count": missing_count,
        "data_quality_tier": "unknown",
        "corpus_tags": case_data.get("corpus_tags") or [],
    }


def structured_case_to_row(case: EvalCase) -> dict[str, Any]:
    gold = case.data.get("gold_output")
    if not isinstance(gold, dict):
        return {}
    split_group = split_group_for_case(case_id=case.case_id, case_data=case.data)
    row = build_row_provenance(
        case_id=case.case_id,
        source="structured_dq_eval",
        source_path=str(STRUCTURED_CASES_DIR / f"{case.case_id}.json"),
        case_data=case.data,
        split=assign_split(split_group),
    )
    row.update(
        {
            "user_text": case.data.get("user_question"),
            "screen_context": None,
            "context_features": _context_features_from_case(case.data),
            **labels_from_structured_dq_gold(gold),
        }
    )
    return row


def chat_case_to_row(case: ChatEvalCase) -> dict[str, Any]:
    split_group = split_group_for_case(case_id=case.case_id, case_data=case.data)
    row = build_row_provenance(
        case_id=case.case_id,
        source="chat_eval",
        source_path=str(CHAT_CASES_DIR / f"{case.case_id}.json"),
        case_data=case.data,
        split=assign_split(split_group),
    )
    row.update(
        {
            "user_text": case.data.get("user_message"),
            "screen_context": case.data.get("screen_context")
            if isinstance(case.data.get("screen_context"), dict)
            else None,
            "context_features": _context_features_from_case(case.data),
            **labels_from_chat_eval(case.data),
        }
    )
    return row


def oc_case_to_row(case_id: str, case_data: dict[str, Any], *, source_path: str) -> dict[str, Any]:
    gold = case_data.get("gold_output")
    if not isinstance(gold, dict):
        return {}
    split_group = split_group_for_case(case_id=case_id, case_data=case_data)
    row = build_row_provenance(
        case_id=case_id,
        source="oc_eval",
        source_path=source_path,
        case_data=case_data,
        split=assign_split(split_group),
    )
    row.update(
        {
            "user_text": case_data.get("user_question"),
            "screen_context": None,
            "context_features": _context_features_from_case(case_data),
            **labels_from_opportunity_candidate_gold(
                gold,
                expected_graduation=case_data.get("expected_graduation"),
            ),
        }
    )
    return row


def export_training_dataset(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    statuses: frozenset[str] = TRAINING_EXPORT_STATUSES,
    include_structured: bool = True,
    include_chat: bool = True,
    include_oc: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status_set = set(statuses)

    if include_structured:
        for structured_case in load_structured_cases(statuses=status_set):
            row = structured_case_to_row(structured_case)
            if row:
                rows.append(row)

    if include_chat:
        for chat_case in load_chat_cases(statuses=status_set):
            if not chat_case.data.get("routing_expectations", {}).get(
                "run_opportunity_preflight"
            ) and not chat_case.data.get("expected_stance"):
                continue
            row = chat_case_to_row(chat_case)
            if row:
                rows.append(row)

    if include_oc:
        for item in _load_oc_cases(statuses=status_set):
            row = oc_case_to_row(item["case_id"], item["data"], source_path=str(item["path"]))
            if row:
                rows.append(row)

    eligible_rows = [row for row in rows if row_is_training_eligible(row, statuses=statuses)]
    leakage = check_split_leakage(eligible_rows)

    version = _now_tag()
    export_dir = output_dir / version
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = export_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in eligible_rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

    split_counts = Counter(str(row.get("split") or "unknown") for row in eligible_rows)
    manifest = {
        "version": version,
        "exported_at": datetime.now(UTC).isoformat(),
        "row_count": len(eligible_rows),
        "raw_row_count": len(rows),
        "dataset_path": str(dataset_path),
        "training_export_statuses": sorted(statuses),
        "split_counts": dict(split_counts),
        "sources": dict(Counter(str(row.get("source") or "unknown") for row in eligible_rows)),
        "leakage_violations": leakage,
        "leakage_check_passed": not leakage,
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    return manifest


def _rows_for_split(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split") or "") == split_name]


def _split_rows(rows: list[dict[str, Any]], *, holdout_ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    holdout = _rows_for_split(rows, "holdout")
    if holdout:
        train = [row for row in rows if row.get("split") in {"train", "validation"}]
        return train, holdout
    if len(rows) < 2:
        return rows, []
    split_at = max(1, int(len(rows) * (1.0 - holdout_ratio)))
    return rows[:split_at], rows[split_at:]


class _ConstantLabelPredictor:
    def __init__(self, value: object):
        self.value = value
        self.classes_ = [value]

    def predict(self, matrix) -> list:
        count = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)
        return [self.value] * count

    def predict_proba(self, matrix):
        import numpy as np

        count = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)
        return np.ones((count, 1), dtype=float)


class SupervisedSynthesisPipeline:
    def __init__(
        self,
        *,
        vectorizer,
        triage_model,
        graduate_model,
        stance_model,
        missing_model,
        missing_vocab: list[str],
    ):
        self.vectorizer = vectorizer
        self.triage_model = triage_model
        self.graduate_model = graduate_model
        self.stance_model = stance_model
        self.missing_model = missing_model
        self.missing_vocab = missing_vocab

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        matrix = self.vectorizer.transform(texts)
        predictions: list[dict[str, Any]] = []
        for idx in range(matrix.shape[0]):
            row = matrix[idx]
            triage_probs = self.triage_model.predict_proba(row)[0]
            triage_idx = int(triage_probs.argmax())
            next_action = str(self.triage_model.classes_[triage_idx])
            confidence = float(triage_probs[triage_idx])
            should_graduate = bool(self.graduate_model.predict(row)[0])
            synthesis_stance = str(self.stance_model.predict(row)[0])
            missing_tags: list[str] = []
            if self.missing_model is not None and self.missing_vocab:
                missing_preds = self.missing_model.predict(row)[0]
                missing_tags = [self.missing_vocab[i] for i, active in enumerate(missing_preds) if active]
            predictions.append(
                {
                    "next_action": next_action,
                    "should_graduate": should_graduate,
                    "synthesis_stance": synthesis_stance,
                    "missing_input_tags": missing_tags,
                    "confidence": confidence,
                }
            )
        return predictions


def _score_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"count": len(rows)}
    if not rows:
        return metrics

    def _acc(label_field: str, pred_field: str) -> float:
        correct = 0
        total = 0
        for row, pred in zip(rows, predictions, strict=False):
            labels = extract_labels_from_row(row)
            if not labels:
                continue
            expected = labels.get(label_field)
            if expected is None:
                continue
            total += 1
            actual = pred.get(pred_field)
            if isinstance(expected, bool):
                if bool(actual) == expected:
                    correct += 1
            elif str(actual) == str(expected):
                correct += 1
        return round(correct / total, 4) if total else 0.0

    metrics["triage_accuracy"] = _acc("next_action", "next_action")
    metrics["graduate_recall"] = _acc("should_graduate", "should_graduate")
    metrics["stance_accuracy"] = _acc("synthesis_stance", "synthesis_stance")
    metrics["fallback_threshold"] = DEFAULT_ROLLOUT_GATES["min_triage_accuracy"]
    return metrics


def train_baseline_classifier(
    *,
    dataset_path: Path | None = None,
    output_dir: Path = DEFAULT_MODEL_DIR,
    holdout_ratio: float = 0.2,
) -> dict[str, Any]:
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    if dataset_path is None:
        manifests = sorted(DEFAULT_OUTPUT_DIR.glob("*/manifest.json"))
        if not manifests:
            raise ValueError("No exported dataset found; run export first")
        manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        dataset_path = Path(str(manifest["dataset_path"]))

    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [row for row in rows if extract_labels_from_row(row)]
    if len(eligible) < 2:
        raise ValueError("Need at least two labeled rows to train the baseline classifier")

    leakage = check_split_leakage(eligible)
    if leakage:
        raise ValueError(f"Split leakage detected for groups: {', '.join(leakage)}")

    train_rows, holdout_rows = _split_rows(eligible, holdout_ratio=holdout_ratio)
    texts = [featurize_context_row(row) for row in train_rows]
    labels = [extract_labels_from_row(row) or {} for row in train_rows]

    triage_labels = [str(item.get("next_action") or "research") for item in labels]
    triage_classes = sorted(set(triage_labels))
    stance_labels = [str(item.get("synthesis_stance") or "unknown") for item in labels]
    stance_classes = sorted(set(stance_labels))
    missing_vocab = sorted(
        {tag for item in labels for tag in item.get("missing_input_tags") or [] if isinstance(tag, str) and tag}
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    x_train = vectorizer.fit_transform(texts)

    triage_model = LogisticRegression(max_iter=1000)
    triage_model.fit(x_train, triage_labels)

    graduate_model = LogisticRegression(max_iter=1000)
    graduate_model.fit(x_train, [bool(item.get("should_graduate")) for item in labels])

    if len(stance_classes) >= 2:
        stance_model = LogisticRegression(max_iter=1000)
        stance_model.fit(x_train, stance_labels)
    else:
        stance_model = _ConstantLabelPredictor(stance_classes[0] if stance_classes else "unknown")

    missing_matrix = np.zeros((len(labels), len(missing_vocab)), dtype=int)
    missing_index = {name: idx for idx, name in enumerate(missing_vocab)}
    for row_idx, item in enumerate(labels):
        for tag in item.get("missing_input_tags") or []:
            if tag in missing_index:
                missing_matrix[row_idx, missing_index[tag]] = 1
    missing_model = OneVsRestClassifier(LogisticRegression(max_iter=1000)) if missing_vocab else None
    if missing_model is not None:
        missing_model.fit(x_train, missing_matrix)

    pipeline = SupervisedSynthesisPipeline(
        vectorizer=vectorizer,
        triage_model=triage_model,
        graduate_model=graduate_model,
        stance_model=stance_model,
        missing_model=missing_model,
        missing_vocab=missing_vocab,
    )

    holdout_predictions = pipeline.predict([featurize_context_row(row) for row in holdout_rows])
    metrics = _score_predictions(holdout_rows, holdout_predictions)
    metrics["train_rows"] = len(train_rows)
    metrics["holdout_rows"] = len(holdout_rows)
    metrics["triage_classes"] = triage_classes
    metrics["stance_classes"] = stance_classes

    version = _now_tag()
    model_dir = output_dir / version
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "model.joblib"
    artifact = {
        "pipeline": pipeline,
        "missing_vocab": missing_vocab,
        "triage_classes": triage_classes,
        "stance_classes": stance_classes,
        "default_confidence": 0.82,
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
    }
    joblib.dump(artifact, artifact_path)

    dataset_manifest = {
        "dataset_path": str(dataset_path),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "leakage_check_passed": not leakage,
    }
    write_model_card(model_dir, metrics=metrics, dataset_manifest=dataset_manifest)

    registry = {
        "active_model_path": _repo_relative_path(artifact_path),
        "active_version": version,
        "updated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "rollout_gates": DEFAULT_ROLLOUT_GATES,
    }
    DEFAULT_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=True, default=str), encoding="utf-8")

    report_path = model_dir / "metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    return {
        "artifact_path": str(artifact_path),
        "registry_path": str(DEFAULT_REGISTRY_PATH),
        "metrics": metrics,
        "model_dir": str(model_dir),
    }


def evaluate_offline(
    *,
    dataset_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    import joblib

    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    holdout_rows = _rows_for_split(rows, "holdout") or rows
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    predictions = pipeline.predict([featurize_context_row(row) for row in holdout_rows])
    metrics = _score_predictions(holdout_rows, predictions)
    by_opportunity_type: dict[str, dict[str, Any]] = {}
    by_failure_type: dict[str, dict[str, Any]] = {}
    for row, pred in zip(holdout_rows, predictions, strict=False):
        opp = str((row.get("context_features") or {}).get("opportunity_type") or "unknown")
        failure = str(row.get("failure_type") or "unknown")
        by_opportunity_type.setdefault(opp, {"rows": [], "preds": []})
        by_failure_type.setdefault(failure, {"rows": [], "preds": []})
        by_opportunity_type[opp]["rows"].append(row)
        by_opportunity_type[opp]["preds"].append(pred)
        by_failure_type[failure]["rows"].append(row)
        by_failure_type[failure]["preds"].append(pred)
    metrics["by_opportunity_type"] = {
        key: _score_predictions(bucket["rows"], bucket["preds"]) for key, bucket in by_opportunity_type.items()
    }
    metrics["by_failure_type"] = {
        key: _score_predictions(bucket["rows"], bucket["preds"]) for key, bucket in by_failure_type.items()
    }
    return metrics


def check_rollout_gates(
    *,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = {**DEFAULT_ROLLOUT_GATES, **(gates or {})}
    checks = {
        "triage_accuracy_pass": metrics.get("triage_accuracy", 0) >= thresholds["min_triage_accuracy"],
        "stance_accuracy_pass": metrics.get("stance_accuracy", 0) >= thresholds["min_stance_accuracy"],
    }
    if baseline_metrics:
        regression = 0
        if metrics.get("graduate_recall", 0) < baseline_metrics.get("graduate_recall", 0):
            regression += 1
        checks["no_metric_regression"] = regression <= thresholds["max_gate_failure_regression"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
    }


def build_supervised_eval_summary(
    *,
    rows: list[dict[str, Any]],
    model_path: Path,
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare supervised predictions to gold labels for eval report attachment."""
    import joblib

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    predictions = pipeline.predict([featurize_context_row(row) for row in rows])
    metrics = _score_predictions(rows, predictions)
    metrics["by_opportunity_type"] = {}
    metrics["by_failure_type"] = {}
    for row, pred in zip(rows, predictions, strict=False):
        opp = str((row.get("context_features") or {}).get("opportunity_type") or "unknown")
        failure = str(row.get("failure_type") or "unknown")
        metrics["by_opportunity_type"].setdefault(opp, {"rows": [], "preds": []})
        metrics["by_failure_type"].setdefault(failure, {"rows": [], "preds": []})
        metrics["by_opportunity_type"][opp]["rows"].append(row)
        metrics["by_opportunity_type"][opp]["preds"].append(pred)
        metrics["by_failure_type"][failure]["rows"].append(row)
        metrics["by_failure_type"][failure]["preds"].append(pred)
    metrics["by_opportunity_type"] = {
        key: _score_predictions(bucket["rows"], bucket["preds"])
        for key, bucket in metrics["by_opportunity_type"].items()
    }
    metrics["by_failure_type"] = {
        key: _score_predictions(bucket["rows"], bucket["preds"]) for key, bucket in metrics["by_failure_type"].items()
    }
    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "rollout_gates": check_rollout_gates(metrics=metrics, baseline_metrics=baseline_metrics),
        "row_count": len(rows),
    }


def rows_from_structured_cases(cases: list[EvalCase]) -> list[dict[str, Any]]:
    return [row for case in cases if (row := structured_case_to_row(case))]


def rows_from_chat_cases(cases: list[ChatEvalCase]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        if not case.data.get("routing_expectations", {}).get("run_opportunity_preflight") and not case.data.get(
            "expected_stance"
        ):
            continue
        row = chat_case_to_row(case)
        if row:
            output.append(row)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQ synthesis supervised training loop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export versioned supervised dataset")
    export_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    export_parser.add_argument("--no-structured", action="store_true")
    export_parser.add_argument("--no-chat", action="store_true")
    export_parser.add_argument("--no-oc", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train baseline supervised classifier")
    train_parser.add_argument("--dataset", default=None)
    train_parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    train_parser.add_argument("--holdout-ratio", type=float, default=0.2)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained artifact offline")
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--baseline-metrics", default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "export":
        manifest = export_training_dataset(
            output_dir=Path(args.output_dir),
            include_structured=not args.no_structured,
            include_chat=not args.no_chat,
            include_oc=not args.no_oc,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=True, default=str))
        return 0
    if args.command == "train":
        result = train_baseline_classifier(
            dataset_path=Path(args.dataset) if args.dataset else None,
            output_dir=Path(args.output_dir),
            holdout_ratio=args.holdout_ratio,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
        return 0
    if args.command == "eval":
        baseline_metrics = None
        if args.baseline_metrics:
            baseline_metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
        metrics = evaluate_offline(dataset_path=Path(args.dataset), model_path=Path(args.model))
        gate_report = check_rollout_gates(metrics=metrics, baseline_metrics=baseline_metrics)
        print(json.dumps({"metrics": metrics, "rollout_gates": gate_report}, indent=2, ensure_ascii=True, default=str))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
