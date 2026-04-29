"""
╔══════════════════════════════════════════════════════════════════════╗
║         VisiHealth AI — Kaggle Test Set Evaluation Notebook         ║
║                                                                      ║
║  HOW TO USE:                                                         ║
║  1. Create a new Kaggle notebook                                     ║
║  2. Add your SLAKE dataset (or the Kaggle public SLAKE dataset)      ║
║  3. Upload best_checkpoint.pth as a Kaggle dataset                   ║
║  4. Paste this entire file into a single code cell and run it        ║
║                                                                      ║
║  Expected runtime: ~3–5 minutes on T4 GPU                           ║
╚══════════════════════════════════════════════════════════════════════╝

STEP 0 — CONFIGURE THESE PATHS BEFORE RUNNING
"""

# ─── ✏️  EDIT THESE TWO PATHS ───────────────────────────────────────────────
SLAKE_ROOT      = "/kaggle/input/datasets/hammad04666/slake-vqa/Slake1.0"          # folder containing train.json, test.json, imgs/
CHECKPOINT_PATH = "/kaggle/input/datasets/hammad04666/visihealthdataset/best_checkpoint.pth"
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================
# CELL 1 — Install missing deps (run once, ~30 s)
# ============================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "torchvision", "Pillow"], check=True)

# ============================================================
# CELL 2 — Imports
# ============================================================
import os, json, math, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

print("✅ Imports OK")
print(f"   PyTorch  : {torch.__version__}")
print(f"   CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU      : {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CELL 3 — Inline Model Code (CNN + BERT + Fusion)
# ============================================================

# ── CNN ──────────────────────────────────────────────────────
from torchvision.models import resnet50, ResNet50_Weights

class ROIAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        attn = self.attention(x)
        return x * attn, attn


class MedicalCNN(nn.Module):
    def __init__(self, dropout=0.25, feature_dim=512, num_classes=202):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.roi_attention_layer3 = ROIAttention(1024)
        self.roi_attention_layer4 = ROIAttention(2048)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_projection = nn.Sequential(
            nn.Linear(2048 + 1024 + 2048, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x, return_attention=False):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        global_f = self.global_pool(x4).squeeze(-1).squeeze(-1)
        roi3, am3 = self.roi_attention_layer3(x3)
        roi3 = self.global_pool(roi3).squeeze(-1).squeeze(-1)
        roi4, am4 = self.roi_attention_layer4(x4)
        roi4 = self.global_pool(roi4).squeeze(-1).squeeze(-1)
        combined = torch.cat([global_f, roi3, roi4], dim=1)
        features = self.feature_projection(combined)
        seg_mask = self.segmentation_head(x4)
        spatial  = x4.flatten(2).transpose(1, 2)  # [B, P, 2048]
        out = {"features": features, "segmentation_mask": seg_mask, "spatial_features": spatial}
        if return_attention:
            out["attention_maps"] = {"layer3": am3, "layer4": am4}
        return out


# ── BERT ─────────────────────────────────────────────────────
BERT_MODEL_NAME = "michiyasunaga/BioLinkBERT-base"

class MedicalBERTEncoder(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1, freeze_layers=2):
        super().__init__()
        print(f"   Loading {BERT_MODEL_NAME} …")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.bert      = AutoModel.from_pretrained(BERT_MODEL_NAME)
        # Freeze bottom layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.projection(cls)

    def tokenize(self, questions, max_length=128):
        return self.tokenizer(questions, padding="max_length", truncation=True,
                              max_length=max_length, return_tensors="pt")


# ── Fusion ───────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim=512, text_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.common_dim = 512
        self.visual_proj         = nn.Linear(visual_dim, self.common_dim)
        self.visual_spatial_proj = nn.Linear(2048, self.common_dim)
        self.text_proj           = nn.Linear(text_dim,   self.common_dim)
        self.cross_attn = nn.MultiheadAttention(self.common_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(self.common_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(self.common_dim * 2, self.common_dim), nn.Sigmoid())
        self.norm_text   = nn.LayerNorm(self.common_dim)
        self.norm_visual = nn.LayerNorm(self.common_dim)
        self.norm_cross  = nn.LayerNorm(self.common_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.common_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
        )
        self.output_dim = 256

    def forward(self, visual_features, text_features, visual_seq=None, text_seq=None):
        v_global = self.visual_proj(visual_features)
        t_cls    = self.text_proj(text_features)
        v_seq = self.norm_visual(self.visual_spatial_proj(visual_seq)) if visual_seq is not None else v_global.unsqueeze(1)
        t_seq = self.norm_text(self.text_proj(text_seq))               if text_seq   is not None else t_cls.unsqueeze(1)
        cross_out, _ = self.cross_attn(t_seq, v_seq, v_seq)
        cross_pooled = self.norm_cross(cross_out.mean(dim=1))
        self_out, _  = self.self_attn(t_seq, t_seq, t_seq)
        self_pooled  = self_out.mean(dim=1)
        gate_val     = self.gate(torch.cat([cross_pooled, self_pooled], dim=1))
        attended     = gate_val * cross_pooled + (1 - gate_val) * self_pooled
        return self.fusion_mlp(torch.cat([attended, v_global], dim=1))


class ROILocalizer(nn.Module):
    def __init__(self, feature_dim=512, num_rois=39):
        super().__init__()
        self.roi_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_rois), nn.Sigmoid()
        )
    def forward(self, x): return self.roi_classifier(x)


CLOSED_ANSWERS = {'yes','no','none','not seen','both','both lungs','a little','much','almost the same'}

class AnswerPredictor(nn.Module):
    def __init__(self, input_dim, num_classes, answer_vocab=None):
        super().__init__()
        self.num_classes = num_classes
        if answer_vocab:
            self.closed_indices = sorted([idx for a,idx in answer_vocab.items() if a in CLOSED_ANSWERS])
            self.open_indices   = sorted([idx for a,idx in answer_vocab.items() if a not in CLOSED_ANSWERS])
        else:
            self.closed_indices, self.open_indices = [], list(range(num_classes))
        nc, no = max(len(self.closed_indices),1), max(len(self.open_indices),1)
        self.closed_head = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,nc))
        self.open_head   = nn.Sequential(nn.Linear(input_dim,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
                                          nn.Linear(512,256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,no))

    def forward(self, x):
        B = x.size(0)
        logits = torch.full((B, self.num_classes), -1e4, device=x.device, dtype=torch.float32)
        if self.closed_indices:
            logits[:, self.closed_indices] = self.closed_head(x).to(torch.float32)
        if self.open_indices:
            logits[:, self.open_indices]   = self.open_head(x).to(torch.float32)
        return logits


class QuestionTypeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,2))
    def forward(self, x): return self.classifier(x)


class VisiHealthModel(nn.Module):
    def __init__(self, cnn, bert, num_classes, answer_vocab=None):
        super().__init__()
        self.cnn  = cnn
        self.bert = bert
        self.fusion           = CrossAttentionFusion()
        self.answer_predictor = AnswerPredictor(256, num_classes, answer_vocab)
        self.qt_classifier    = QuestionTypeClassifier(256)
        self.roi_localizer    = ROILocalizer(512, num_rois=39)

    def forward(self, images, input_ids, attention_mask):
        cnn_out        = self.cnn(images)
        visual_f       = cnn_out["features"]
        visual_spatial = cnn_out.get("spatial_features")
        bert_all       = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_f         = self.bert.projection(bert_all.last_hidden_state[:, 0, :])
        text_seq       = bert_all.last_hidden_state
        fused = self.fusion(visual_f, text_f, visual_seq=visual_spatial, text_seq=text_seq)
        return {
            "answer_logits": self.answer_predictor(fused),
            "qt_logits":     self.qt_classifier(fused),
            "roi_scores":    self.roi_localizer(visual_f),
        }

print("✅ Model classes defined")

# ============================================================
# CELL 4 — Dataset
# ============================================================
IMG_SIZE = 336

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class SLAKEEval(Dataset):
    """Minimal SLAKE test loader — uses the training vocab so labels align."""
    def __init__(self, slake_root, answer_vocab, split="test"):
        self.root = Path(slake_root)
        self.answer_vocab = answer_vocab       # {answer_text: idx}
        qa_path = self.root / f"{split}.json"
        raw = json.loads(qa_path.read_text(encoding="utf-8"))
        # Filter to English + answers in vocab
        self.samples = [
            q for q in raw
            if q.get("q_lang", "en") == "en" and q["answer"] in answer_vocab
        ]
        print(f"   {split} set: {len(self.samples)} English samples with known answers")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        img_path = self.root / "imgs" / s["img_name"]
        image  = Image.open(img_path).convert("RGB")
        image  = TRANSFORM(image)
        answer = self.answer_vocab[s["answer"]]
        return {"image": image, "answer": torch.tensor(answer, dtype=torch.long),
                "question": s["question"], "img_name": s["img_name"]}

# ============================================================
# CELL 5 — Build vocab from training set
# ============================================================
print("\n📂 Building answer vocab from training set …")
train_path = Path(SLAKE_ROOT) / "train.json"
train_raw  = json.loads(train_path.read_text(encoding="utf-8"))
train_en   = [q for q in train_raw if q.get("q_lang","en") == "en"]

answer_vocab: dict[str, int] = {}
for q in train_en:
    ans = q["answer"]
    if ans not in answer_vocab:
        answer_vocab[ans] = len(answer_vocab)

num_classes = len(answer_vocab)
print(f"   Vocab size: {num_classes} answers")

# ============================================================
# CELL 6 — Load checkpoint & build model
# ============================================================
print("\n🔧 Building model …")
cnn  = MedicalCNN(dropout=0.25, feature_dim=512, num_classes=num_classes)
bert = MedicalBERTEncoder(hidden_size=768, dropout=0.1, freeze_layers=2)

# Override num_classes in CNN feature projection
model = VisiHealthModel(cnn, bert, num_classes=num_classes, answer_vocab=answer_vocab)

print(f"\n📦 Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"   Checkpoint epoch  : {ckpt.get('epoch', 'unknown')}")
print(f"   Saved val accuracy: {ckpt.get('best_val_acc', ckpt.get('val_acc', 'N/A')):.4f}%" 
      if isinstance(ckpt.get('best_val_acc', ckpt.get('val_acc')), float) else 
      f"   Saved val accuracy: {ckpt.get('best_val_acc', 'N/A')}")

model = model.to(device)
model.eval()
print("✅ Model ready")

# ============================================================
# CELL 7 — Run test evaluation
# ============================================================
print("\n🧪 Building test dataloader …")
test_dataset = SLAKEEval(SLAKE_ROOT, answer_vocab, split="test")
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=torch.cuda.is_available())

print(f"\n🚀 Running evaluation on {len(test_dataset)} test samples …")
correct = 0
total   = 0

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images       = batch["image"].to(device)
        input_ids    = bert.tokenizer(
            list(batch["question"]),
            padding="max_length", truncation=True,
            max_length=128, return_tensors="pt"
        )["input_ids"].to(device)
        attention_mask = bert.tokenizer(
            list(batch["question"]),
            padding="max_length", truncation=True,
            max_length=128, return_tensors="pt"
        )["attention_mask"].to(device)
        answers      = batch["answer"].to(device)

        outputs = model(images, input_ids, attention_mask)
        preds   = outputs["answer_logits"].argmax(dim=1)
        correct += preds.eq(answers).sum().item()
        total   += answers.size(0)

        if (i + 1) % 10 == 0:
            running_acc = 100.0 * correct / total
            print(f"   Batch {i+1}/{len(test_loader)}  |  Running accuracy: {running_acc:.2f}%")

test_accuracy = 100.0 * correct / total

# ============================================================
# CELL 8 — Results
# ============================================================
print("\n" + "="*60)
print("           VisiHealth AI — TEST SET RESULTS")
print("="*60)
print(f"  Total test samples : {total}")
print(f"  Correct predictions: {correct}")
print(f"  ✅ TEST ACCURACY   : {test_accuracy:.4f}%")
print("="*60)

# Save result
result = {
    "test_accuracy": test_accuracy,
    "test_accuracy_status": "evaluated",
    "correct": correct,
    "total": total,
    "checkpoint_epoch": ckpt.get("epoch"),
    "last_known_val_accuracy": ckpt.get("best_val_acc"),
}
out_path = "/kaggle/working/visihealth_test_results.json"
Path(out_path).write_text(json.dumps(result, indent=2))
print(f"\n📄 Results saved to {out_path}")
print("    → Download this file and update your results/VisiHealth_Results.json")
