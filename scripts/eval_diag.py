import json
import traceback
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import get_dataloader
from models import get_cnn_model, get_bert_model, build_visihealth_model

out = {"ok": False}

try:
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PROJECT_ROOT / "checkpoints" / "best_checkpoint.pth", map_location=device)

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

    out["ok"] = True
    out["correct"] = correct
    out["total"] = total
    out["test_accuracy"] = (100.0 * correct / total) if total else 0.0
except Exception as exc:
    out["error"] = repr(exc)
    out["traceback"] = traceback.format_exc()

(PROJECT_ROOT / "eval_diag_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("Wrote eval_diag_result.json")
