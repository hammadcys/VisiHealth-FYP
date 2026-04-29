"""
Refresh model artifacts from current checkpoint and dataset.
- Evaluates test accuracy for best checkpoint
- Regenerates results/VisiHealth_Results.json
- Regenerates results/VisiHealth_Model_Info.json
"""

import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import get_dataloader
from data.dataset import SLAKEDataset
from models import get_cnn_model, get_bert_model, build_visihealth_model


def _load_filtered_split(data_dir: str, split: str = "train", language: str = "en") -> list:
    split_path = Path(data_dir) / f"{split}.json"
    with split_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return [item for item in records if item.get("q_lang", "en") == language]


def _build_normalized_answer_vocab(train_records: list) -> Dict[str, int]:
    answers = set()
    for item in train_records:
        normalized = SLAKEDataset._normalize_answer(item.get("answer", ""))
        if normalized:
            answers.add(normalized)
    return {ans: idx for idx, ans in enumerate(sorted(answers))}


def _evaluate_if_possible(config: dict, checkpoint: dict, device: torch.device) -> Optional[Tuple[float, int, int, Dict[str, int], int]]:
    """
    Try full evaluation path. Returns None when unavailable in this environment
    (e.g., model/tokenizer download/SSL failures).
    """
    try:
        train_loader, train_dataset = get_dataloader(
            data_dir=config["dataset"]["root_dir"],
            split="train",
            batch_size=config["training"]["batch_size"],
            num_workers=0,
            tokenizer_name=config["model"]["bert"]["model_name"],
        )

        test_loader, _ = get_dataloader(
            data_dir=config["dataset"]["root_dir"],
            split="test",
            batch_size=config["training"]["batch_size"],
            num_workers=0,
            tokenizer_name=config["model"]["bert"]["model_name"],
            train_vocab=train_dataset.answer_vocab,
        )

        num_classes = train_dataset.num_classes
        config["model"]["cnn"]["num_classes"] = num_classes

        cnn = get_cnn_model(config).to(device)
        bert = get_bert_model(config).to(device)
        model = build_visihealth_model(config, cnn, bert, answer_vocab=train_dataset.answer_vocab).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                answers = batch["answer"].to(device)

                outputs = model(images, input_ids, attention_mask)
                preds = outputs["answer_logits"].argmax(dim=1)

                correct += preds.eq(answers).sum().item()
                total += answers.size(0)

        test_accuracy = (100.0 * correct / total) if total else 0.0
        return test_accuracy, correct, total, train_dataset.answer_vocab, len(train_dataset)
    except Exception as exc:
        print(f"Evaluation skipped: {exc}")
        return None


def main() -> None:
    project_root = PROJECT_ROOT
    config_path = project_root / "config.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = project_root / "checkpoints" / "best_checkpoint.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    train_records = _load_filtered_split(config["dataset"]["root_dir"], split="train", language="en")
    fallback_answer_vocab = _build_normalized_answer_vocab(train_records)
    fallback_num_classes = len(fallback_answer_vocab)

    old_test_accuracy = None
    old_results_path = project_root / "results" / "VisiHealth_Results.json"
    if old_results_path.exists():
        with old_results_path.open("r", encoding="utf-8") as f:
            old_results = json.load(f)
        old_test_accuracy = old_results.get("test_accuracy")

    evaluation = _evaluate_if_possible(config, checkpoint, device)
    if evaluation is not None:
        test_accuracy, correct, total, answer_vocab, train_sample_count = evaluation
        evaluation_status = "computed"
        notes = "Metrics computed successfully in current environment."
    else:
        test_accuracy = None
        correct = None
        total = None
        answer_vocab = fallback_answer_vocab
        train_sample_count = len(train_records)
        evaluation_status = "pending_re_evaluation_on_current_environment"
        notes = (
            "Checkpoint/config upgraded; local test evaluation unavailable in current environment "
            "(likely HuggingFace download/SSL issue). Re-run on Kaggle or machine with cache/network."
        )

    num_classes = len(answer_vocab)

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "test_accuracy": test_accuracy,
        "test_accuracy_status": evaluation_status,
        "last_known_test_accuracy": old_test_accuracy,
        "correct": correct,
        "total": total,
        "checkpoint": "checkpoints/best_checkpoint.pth",
        "platform": "Local",
        "notes": notes,
    }

    model_info_payload = {
        "model_type": "VisiHealth AI",
        "checkpoint_path": "checkpoints/best_checkpoint.pth",
        "test_accuracy": test_accuracy,
        "test_accuracy_status": evaluation_status,
        "last_known_test_accuracy": old_test_accuracy,
        "num_classes": num_classes,
        "answer_vocab": {str(idx): ans for ans, idx in answer_vocab.items()},
        "image_size": int(config.get("image", {}).get("size", 224)),
        "bert_model": config.get("model", {}).get("bert", {}).get("model_name", "unknown"),
        "usage": {
            "input": f"Medical image ({config.get('image', {}).get('size', 224)}x{config.get('image', {}).get('size', 224)}) + question text",
            "output": "Answer + confidence + ROI scores + rationale",
        },
        "training_info": {
            "dataset": config.get("dataset", {}).get("name", "SLAKE"),
            "total_samples": train_sample_count,
            "epochs": int(checkpoint.get("epoch", -1)) + 1,
            "best_val_acc": float(checkpoint.get("best_val_acc", 0.0)),
            "best_val_loss": float(checkpoint.get("best_val_loss", 0.0)),
            "platform": "Local",
            "fusion_method": config.get("model", {}).get("fusion", {}).get("method", "concat"),
            "freeze_layers": config.get("model", {}).get("bert", {}).get("freeze_layers", None),
            "label_smoothing": config.get("training", {}).get("label_smoothing", None),
        },
    }

    results_path = results_dir / "VisiHealth_Results.json"
    model_info_path = results_dir / "VisiHealth_Model_Info.json"

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    with model_info_path.open("w", encoding="utf-8") as f:
        json.dump(model_info_payload, f, indent=2)

    print("Artifacts refreshed successfully")
    print(f"Evaluation Status: {evaluation_status}")
    if test_accuracy is not None:
        print(f"Test Accuracy: {test_accuracy:.4f}%")
        print(f"Correct/Total: {correct}/{total}")
    else:
        print("Test Accuracy: pending")
        print(f"Last Known Test Accuracy: {old_test_accuracy}")
    print(f"Saved: {results_path}")
    print(f"Saved: {model_info_path}")


if __name__ == "__main__":
    main()
