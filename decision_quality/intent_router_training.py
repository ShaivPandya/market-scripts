"""Export, train, and evaluate supervised intent-router models."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_quality.chat_eval_runner import CASES_DIR, ChatEvalCase, load_cases
from decision_quality.intent_router_supervised import extract_label_from_row, featurize_training_row, write_model_card

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "intent_router_training"
DEFAULT_MODEL_DIR = ROOT / "data" / "intent_router_models"
DEFAULT_REGISTRY_PATH = DEFAULT_MODEL_DIR / "registry.json"


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _dedupe_key(row: dict[str, Any]) -> str:
    session_id = str(row.get("session_id") or "")
    client_turn_id = str(row.get("client_turn_id") or "")
    if session_id and client_turn_id:
        return f"{session_id}:{client_turn_id}"
    payload = json.dumps(
        {
            "user_text": row.get("user_text"),
            "screen_context": row.get("screen_context"),
            "regex_baseline": row.get("regex_baseline"),
        },
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eval_fixture_to_training_row(case: ChatEvalCase) -> dict[str, Any]:
    screen_context = case.data.get("screen_context")
    routing_expectations = case.data.get("routing_expectations") or {}
    label_tools = routing_expectations.get("required_tool_names") or case.data.get("expected_tool_names") or []
    return {
        "row_id": f"fixture:{case.case_id}",
        "source": "eval_fixture",
        "case_id": case.case_id,
        "user_text": case.data.get("user_message"),
        "screen_context": screen_context if isinstance(screen_context, dict) else None,
        "recent_session_features": [],
        "routing_expectations": routing_expectations,
        "label_intent_class": routing_expectations.get("intent_class"),
        "label_run_hidden_dq": routing_expectations.get("run_hidden_dq"),
        "label_run_opportunity_preflight": routing_expectations.get("run_opportunity_preflight"),
        "label_workflow_name": routing_expectations.get("workflow_name"),
        "label_tool_names": label_tools,
        "label_reviewer": "eval_fixture",
        "labeled_at": datetime.now(UTC).isoformat(),
    }


def export_training_dataset(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    include_fixtures: bool = True,
    fixture_prefix: str | None = "routing_",
    include_db_rows: bool = True,
    labeled_only: bool = False,
    limit: int = 5000,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if include_db_rows:
        from api.intent_router_training_store import list_training_rows

        rows.extend(list_training_rows(limit=limit, labeled_only=labeled_only))

    if include_fixtures:
        selectors = None
        if fixture_prefix:
            selectors = [path.stem for path in sorted(CASES_DIR.glob("*.json")) if path.stem.startswith(fixture_prefix)]
        for case in load_cases(case_selectors=selectors, statuses={"review", "approved", "draft"}):
            rows.append(eval_fixture_to_training_row(case))

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[_dedupe_key(row)] = row
    dataset_rows = list(deduped.values())

    version = _now_tag()
    export_dir = output_dir / version
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = export_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in dataset_rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

    label_coverage = sum(1 for row in dataset_rows if extract_label_from_row(row))
    manifest = {
        "version": version,
        "exported_at": datetime.now(UTC).isoformat(),
        "row_count": len(dataset_rows),
        "labeled_row_count": label_coverage,
        "dataset_path": str(dataset_path),
        "sources": Counter(str(row.get("source") or "telemetry") for row in dataset_rows),
    }
    manifest["sources"] = dict(manifest["sources"])
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    return manifest


def _split_rows(rows: list[dict[str, Any]], *, holdout_ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    if len(rows) < 2:
        return rows, []
    split_at = max(1, int(len(rows) * (1.0 - holdout_ratio)))
    return rows[:split_at], rows[split_at:]


class _ConstantLabelPredictor:
    def __init__(self, value: str):
        self.value = value
        self.classes_ = [value]

    def predict(self, matrix) -> list[str]:
        count = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)
        return [self.value] * count

    def predict_proba(self, matrix):
        import numpy as np

        count = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)
        return np.ones((count, 1), dtype=float)


class SupervisedRouterPipeline:
    def __init__(
        self,
        *,
        vectorizer,
        intent_model,
        hidden_dq_model,
        opportunity_model,
        workflow_model,
        tool_model,
        tool_vocab: list[str],
    ):
        self.vectorizer = vectorizer
        self.intent_model = intent_model
        self.hidden_dq_model = hidden_dq_model
        self.opportunity_model = opportunity_model
        self.workflow_model = workflow_model
        self.tool_model = tool_model
        self.tool_vocab = tool_vocab

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        matrix = self.vectorizer.transform(texts)
        predictions: list[dict[str, Any]] = []
        for idx in range(matrix.shape[0]):
            row = matrix[idx]
            intent_probs = self.intent_model.predict_proba(row)[0]
            intent_idx = int(intent_probs.argmax())
            intent_class = self.intent_model.classes_[intent_idx]
            confidence = float(intent_probs[intent_idx])
            workflow_name = self.workflow_model.predict(row)[0]
            tool_names: list[str] = []
            if self.tool_model is not None and self.tool_vocab:
                tool_preds = self.tool_model.predict(row)[0]
                tool_names = [self.tool_vocab[i] for i, active in enumerate(tool_preds) if active]
            predictions.append(
                {
                    "intent_class": str(intent_class),
                    "run_hidden_dq": bool(self.hidden_dq_model.predict(row)[0]),
                    "run_opportunity_preflight": bool(self.opportunity_model.predict(row)[0]),
                    "workflow_name": None if workflow_name == "__none__" else str(workflow_name),
                    "tool_names": tool_names,
                    "confidence": confidence,
                    "tool_pack": str(intent_class),
                }
            )
        return predictions


def _score_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"count": len(rows)}
    if not rows:
        return metrics

    def _acc(field: str) -> float:
        correct = 0
        total = 0
        for row, pred in zip(rows, predictions, strict=False):
            label = extract_label_from_row(row)
            if not label or label.get(field) is None:
                continue
            total += 1
            if (
                bool(pred.get(field)) == bool(label.get(field))
                if field.startswith("run_")
                else pred.get(field) == label.get(field)
            ):
                correct += 1
        return round(correct / total, 4) if total else 0.0

    metrics["intent_accuracy"] = _acc("intent_class")
    metrics["hidden_dq_recall"] = _acc("run_hidden_dq")
    metrics["opportunity_preflight_recall"] = _acc("run_opportunity_preflight")
    metrics["workflow_accuracy"] = _acc("workflow_name")
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
    labeled_rows = [row for row in rows if extract_label_from_row(row)]
    if len(labeled_rows) < 2:
        raise ValueError("Need at least two labeled rows to train the baseline classifier")

    train_rows, holdout_rows = _split_rows(labeled_rows, holdout_ratio=holdout_ratio)
    texts = [featurize_training_row(row) for row in train_rows]
    labels = [extract_label_from_row(row) or {} for row in train_rows]

    intent_classes = sorted({str(item.get("intent_class")) for item in labels if item.get("intent_class")})
    tool_vocab = sorted({tool for item in labels for tool in item.get("tool_names") or []})
    workflow_vocab = sorted({str(item.get("workflow_name")) for item in labels if item.get("workflow_name")})

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    x_train = vectorizer.fit_transform(texts)

    intent_model = LogisticRegression(max_iter=1000)
    intent_model.fit(x_train, [str(item.get("intent_class") or "general_research") for item in labels])

    hidden_dq_model = LogisticRegression(max_iter=1000)
    hidden_dq_model.fit(x_train, [bool(item.get("run_hidden_dq")) for item in labels])

    opportunity_model = LogisticRegression(max_iter=1000)
    opportunity_model.fit(x_train, [bool(item.get("run_opportunity_preflight")) for item in labels])

    workflow_labels = [str(item.get("workflow_name") or "__none__") for item in labels]
    unique_workflows = sorted(set(workflow_labels))
    if len(unique_workflows) >= 2:
        workflow_model = LogisticRegression(max_iter=1000)
        workflow_model.fit(x_train, workflow_labels)
    else:
        workflow_model = _ConstantLabelPredictor(unique_workflows[0])

    tool_matrix = np.zeros((len(labels), len(tool_vocab)), dtype=int)
    tool_index = {name: idx for idx, name in enumerate(tool_vocab)}
    for row_idx, item in enumerate(labels):
        for tool in item.get("tool_names") or []:
            if tool in tool_index:
                tool_matrix[row_idx, tool_index[tool]] = 1
    tool_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    if tool_vocab:
        tool_model.fit(x_train, tool_matrix)

    pipeline = SupervisedRouterPipeline(
        vectorizer=vectorizer,
        intent_model=intent_model,
        hidden_dq_model=hidden_dq_model,
        opportunity_model=opportunity_model,
        workflow_model=workflow_model,
        tool_model=tool_model if tool_vocab else None,
        tool_vocab=tool_vocab,
    )

    holdout_predictions = pipeline.predict([featurize_training_row(row) for row in holdout_rows])
    metrics = _score_predictions(holdout_rows, holdout_predictions)
    metrics["train_rows"] = len(train_rows)
    metrics["holdout_rows"] = len(holdout_rows)
    metrics["fallback_threshold"] = 0.70

    version = _now_tag()
    model_dir = output_dir / version
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "model.joblib"
    artifact = {
        "pipeline": pipeline,
        "tool_vocab": tool_vocab,
        "workflow_vocab": workflow_vocab,
        "intent_classes": intent_classes,
        "default_confidence": 0.82,
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
    }
    joblib.dump(artifact, artifact_path)

    dataset_manifest = {
        "dataset_path": str(dataset_path),
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
    }
    write_model_card(model_dir, metrics=metrics, dataset_manifest=dataset_manifest)

    registry = {
        "active_model_path": str(artifact_path),
        "active_version": version,
        "updated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intent router supervised training loop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export versioned training dataset")
    export_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    export_parser.add_argument("--fixture-prefix", default="routing_")
    export_parser.add_argument("--no-fixtures", action="store_true")
    export_parser.add_argument("--no-db", action="store_true")
    export_parser.add_argument("--labeled-only", action="store_true")
    export_parser.add_argument("--limit", type=int, default=5000)

    train_parser = subparsers.add_parser("train", help="Train baseline supervised classifier")
    train_parser.add_argument("--dataset", default=None)
    train_parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    train_parser.add_argument("--holdout-ratio", type=float, default=0.2)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "export":
        manifest = export_training_dataset(
            output_dir=Path(args.output_dir),
            include_fixtures=not args.no_fixtures,
            fixture_prefix=args.fixture_prefix or None,
            include_db_rows=not args.no_db,
            labeled_only=args.labeled_only,
            limit=args.limit,
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
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
