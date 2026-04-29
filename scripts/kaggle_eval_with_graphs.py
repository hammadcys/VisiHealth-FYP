"""
╔══════════════════════════════════════════════════════════════════════╗
║    VisiHealth AI — Kaggle Eval + Graph Generation (Confusion Matrix, Precision, Recall) ║
║                                                                      ║
║  HOW TO USE:                                                         ║
║  1. Create a new Kaggle notebook with GPU T4 x2                      ║
║  2. Add your SLAKE dataset and best_checkpoint.pth dataset           ║
║  3. Paste this code and run it                                       ║
║  4. It will save the graphs as PNG images to /kaggle/working/        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─── ✏️  EDIT THESE TWO PATHS ───────────────────────────────────────────────
SLAKE_ROOT      = "/kaggle/input/datasets/hammad04666/slake-vqa/Slake1.0"          
CHECKPOINT_PATH = "/kaggle/input/datasets/hammad04666/visihealthdataset/best_checkpoint.pth"
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================
# CELL 1 — Install missing deps
# ============================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "torchvision", "Pillow", "matplotlib", "seaborn", "scikit-learn"], check=True)

# ============================================================
# CELL 2 — Imports
# ============================================================
import os, json, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CELL 3 — Model Code (Condensed for brevity - same as previous)
# ============================================================
from torchvision.models import resnet50, ResNet50_Weights

class ROIAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1), nn.ReLU(inplace=True), nn.Conv2d(in_channels // 4, 1, 1), nn.Sigmoid())
    def forward(self, x): return x * self.attention(x), self.attention(x)

class MedicalCNN(nn.Module):
    def __init__(self, feature_dim=512, num_classes=202):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2; self.layer3 = resnet.layer3; self.layer4 = resnet.layer4
        self.roi_attention_layer3 = ROIAttention(1024)
        self.roi_attention_layer4 = ROIAttention(2048)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_projection = nn.Sequential(nn.Linear(2048 + 1024 + 2048, feature_dim), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(feature_dim, feature_dim))
        self.segmentation_head = nn.Sequential(nn.Conv2d(2048, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 1, 1))
    def forward(self, x):
        x0 = self.layer0(x); x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        global_f = self.global_pool(x4).squeeze(-1).squeeze(-1)
        roi3, _ = self.roi_attention_layer3(x3); roi3 = self.global_pool(roi3).squeeze(-1).squeeze(-1)
        roi4, _ = self.roi_attention_layer4(x4); roi4 = self.global_pool(roi4).squeeze(-1).squeeze(-1)
        features = self.feature_projection(torch.cat([global_f, roi3, roi4], dim=1))
        return {"features": features, "spatial_features": x4.flatten(2).transpose(1, 2)}

class MedicalBERTEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
        self.bert      = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
    def forward(self, input_ids, attention_mask): return self.projection(self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])

class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_proj = nn.Linear(512, 512); self.visual_spatial_proj = nn.Linear(2048, 512); self.text_proj = nn.Linear(768, 512)
        self.cross_attn = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True); self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(1024, 512), nn.Sigmoid()); self.norm_text = nn.LayerNorm(512); self.norm_visual = nn.LayerNorm(512); self.norm_cross = nn.LayerNorm(512)
        self.fusion_mlp = nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
    def forward(self, vf, tf, v_seq, t_seq):
        vg = self.visual_proj(vf)
        vs = self.norm_visual(self.visual_spatial_proj(v_seq)) if v_seq is not None else vg.unsqueeze(1)
        ts = self.norm_text(self.text_proj(t_seq))
        co, _ = self.cross_attn(ts, vs, vs); cp = self.norm_cross(co.mean(dim=1))
        so, _ = self.self_attn(ts, ts, ts); sp = so.mean(dim=1)
        gv = self.gate(torch.cat([cp, sp], dim=1))
        return self.fusion_mlp(torch.cat([gv * cp + (1 - gv) * sp, vg], dim=1))

class AnswerPredictor(nn.Module):
    def __init__(self, nc, vocab):
        super().__init__()
        self.nc = nc
        self.closed_indices = sorted([idx for a,idx in vocab.items() if a in {'yes','no','none','not seen','both','both lungs','a little','much','almost the same'}])
        self.open_indices   = sorted([idx for a,idx in vocab.items() if a not in {'yes','no','none','not seen','both','both lungs','a little','much','almost the same'}])
        self.closed_head = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,max(len(self.closed_indices),1)))
        self.open_head   = nn.Sequential(nn.Linear(256,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512,256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256,max(len(self.open_indices),1)))
    def forward(self, x):
        l = torch.full((x.size(0), self.nc), -1e4, device=x.device, dtype=torch.float32)
        if self.closed_indices: l[:, self.closed_indices] = self.closed_head(x).to(torch.float32)
        if self.open_indices:   l[:, self.open_indices]   = self.open_head(x).to(torch.float32)
        return l

class QuestionTypeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,2))
    def forward(self, x): return self.classifier(x)

class ROILocalizer(nn.Module):
    def __init__(self, feature_dim=512, num_rois=39):
        super().__init__()
        self.roi_classifier = nn.Sequential(nn.Linear(feature_dim,256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256,num_rois), nn.Sigmoid())
    def forward(self, x): return self.roi_classifier(x)

class VisiHealthModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.cnn = MedicalCNN(); self.bert = MedicalBERTEncoder()
        self.fusion = CrossAttentionFusion()
        self.answer_predictor = AnswerPredictor(len(vocab), vocab)
        self.qt_classifier    = QuestionTypeClassifier(256)
        self.roi_localizer    = ROILocalizer(512, num_rois=39)
    def forward(self, img, ids, mask):
        cout = self.cnn(img); bout = self.bert.bert(input_ids=ids, attention_mask=mask)
        f = self.fusion(cout["features"], self.bert.projection(bout.last_hidden_state[:,0,:]), cout["spatial_features"], bout.last_hidden_state)
        return self.answer_predictor(f)

# ============================================================
# CELL 4 — Dataset & Setup
# ============================================================
print("\n📂 Loading exact 202-class answer vocab from training phase …")
idx_to_answer = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "a little", 8: "abdomen", 9: "above the rectum",
    10: "absorb nutrients, digest food, secrete enzymes", 11: "absorb water, digest food, excrete body waste",
    12: "absorb water, excrete body waste", 13: "almost the same", 14: "around the bladder", 15: "arrhythmia, chest tightness",
    16: "atelectasis", 17: "atelectasis, effusion", 18: "atelectasis, mass", 19: "avoid strenuous exercise, quit smoking",
    20: "bacterial infection", 21: "bacterial infection, etc, inflammation", 22: "biotransformation, detoxification",
    23: "black", 24: "black hollow", 25: "bladder", 26: "body", 27: "both", 28: "both lungs", 29: "bottom", 30: "brain",
    31: "brain edema", 32: "brain edema, brain enhancing tumor", 33: "brain edema, brain enhancing tumor, brain non-enhancing tumor",
    34: "brain edema, brain non-enhancing tumor", 35: "brain edema, brain tumor", 36: "brain embryonic tissue dysplasia, chemical factors, genetic factors",
    37: "breathe", 38: "breathing, control heartbeat", 39: "bronchial obstruction", 40: "bullae, chest injury, lung disease",
    41: "cardiomegaly", 42: "cardiomegaly, effusion, infiltration", 43: "cardiomegaly, pneumothorax", 44: "center",
    45: "center, left lung", 46: "cerebral hypoxia, cerebrovascular lesions, craniocerebral injury, intracranial inflammation, intracranial space-occupying lesions",
    47: "chest", 48: "chest pain, chest tightness, dyspnea", 49: "chest pain, cough, cough foamy mucus sputum, dyspnea",
    50: "chest pain, cough, expectoration", 51: "chest pain, dyspnea", 52: "chest pain, dyspnea, hemoptysis",
    53: "chest tightness, difficulty breathing, dry cough, shortness of breath", 54: "chest tightness, fatigue",
    55: "chewing, cutting, maintaining facial contour and assisting pronunciation", 56: "chronic irritation, pulmonary infection",
    57: "circular", 58: "colon", 59: "colon, rectum, small bowel", 60: "colon, small bowel", 61: "colon, stomach",
    62: "confusion, encephalomalacia, increased intracranial pressure, local edema", 63: "coronal plane", 64: "ct",
    65: "dark gray", 66: "digest food, secrete enzymes", 67: "dilated cardiomyopathy, hypertension", 68: "duodenum",
    69: "duodenum, small bowel", 70: "effusion", 71: "enhance physical fitness, live healthy", 72: "enhance physical fitness, quit smoking",
    73: "esophagus", 74: "etc, inflammation, malignant tumor, trauma", 75: "excrete feces, store feces", 76: "eyes",
    77: "eyes, temporal lobe", 78: "gas delivery", 79: "gray", 80: "gray ball on the left", 81: "grey circle on the left",
    82: "head", 83: "heart", 84: "heart, liver", 85: "heart, liver, lung", 86: "heart, liver, lung, spleen",
    87: "heart, liver, spleen", 88: "heart, lung", 89: "hyperdense", 90: "hypodense", 91: "improve the body's immunity",
    92: "increased intracranial pressure, tinnitus, visual impairment, vomiting", 93: "infiltration", 94: "intestine",
    95: "irregular", 96: "keep healthy", 97: "keep healthy, treat brain diseases promptly", 98: "kidney",
    99: "kidney cancer, liver cancer", 100: "kidney, liver", 101: "large bowel", 102: "large bowel, stomach",
    103: "larynx", 104: "left", 105: "left femoral head", 106: "left kidney", 107: "left lobe", 108: "left lung",
    109: "left lung, lower right", 110: "left lung, right", 111: "left lung, upper right", 112: "left temporal lobe",
    113: "left, liver", 114: "left, right lung", 115: "left, top", 116: "light grey", 117: "liver", 118: "liver cancer",
    119: "liver, lung", 120: "liver, stomach", 121: "liver, top", 122: "lower left", 123: "lower left chest",
    124: "lower left lobe", 125: "lower left lung", 126: "lower left, right lung", 127: "lower middle", 128: "lower right",
    129: "lower right lobe", 130: "lower right lung", 131: "lung", 132: "lung cancer", 133: "lung tumor, pulmonary infection, tuberculosis and other diseases",
    134: "mandible", 135: "mandible, parotid", 136: "mass", 137: "medical therapy, supportive therapy", 138: "medical treatment",
    139: "medical treatment, supportive treatment, surgical treatment", 140: "medical treatment, surgical treatment",
    141: "medication, physical therapy", 142: "memory, participate in hearing, speech", 143: "mri", 144: "much",
    145: "neck", 146: "no", 147: "nodule", 148: "none", 149: "not seen", 150: "oval", 151: "parotid",
    152: "pay attention to dietary hygiene, strengthen physical fitness and avoid brain trauma",
    153: "pay attention to prevent cold and keep warm, enhance physical fitness", 154: "pelvic cavity",
    155: "pharmacotherapy, rehabilitation", 156: "physical therapy, surgical treatment", 157: "pleural effusion",
    158: "pneumonia", 159: "pneumothorax", 160: "promote blood flow", 161: "pronunciation, ventilation",
    162: "pulmonary bronchus", 163: "pulmonary infiltration", 164: "pulmonary mass", 165: "pulmonary nodule",
    166: "rectum", 167: "rectum, small bowel", 168: "respiratory system", 169: "right", 170: "right chest",
    171: "right kidney", 172: "right lobe", 173: "right lung", 174: "right lung, upper left", 175: "small bowel",
    176: "spinal cord", 177: "spleen", 178: "stomach", 179: "store urine", 180: "symmetrical to the bone marrow",
    181: "symmetrical to the bottom spine", 182: "symmetrical to the spine", 183: "t1", 184: "t2",
    185: "temporal lobe", 186: "tooth", 187: "top", 188: "trachea", 189: "transverse plane", 190: "u-shaped",
    191: "under the trachea", 192: "upper left", 193: "upper left lobe", 194: "upper left lung",
    195: "upper left of spleen", 196: "upper right", 197: "upper right lobe", 198: "upper right lung",
    199: "white", 200: "x-ray", 201: "yes"
}
answer_vocab = {v: k for k, v in idx_to_answer.items()}

model = VisiHealthModel(answer_vocab).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)["model_state_dict"])
model.eval()

class SLAKEEval(Dataset):
    def __init__(self, root, vocab):
        self.root = Path(root); self.v = vocab
        raw = json.loads((Path(root)/"test.json").read_text(encoding="utf-8"))
        self.s = []
        for q in raw:
            if q.get("q_lang", "en") == "en":
                ans = str(q["answer"]).lower().strip()
                if ans.endswith('.'): ans = ans[:-1]
                if ans in vocab:
                    q["answer"] = ans
                    self.s.append(q)
    def __len__(self): return len(self.s)
    def __getitem__(self, idx):
        return {"img": transforms.Compose([transforms.Resize((336, 336)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])(Image.open(self.root/"imgs"/self.s[idx]["img_name"]).convert("RGB")),
                "ans": self.v[self.s[idx]["answer"]], "q": self.s[idx]["question"]}

test_loader = DataLoader(SLAKEEval(SLAKE_ROOT, answer_vocab), batch_size=16, shuffle=False)

# ============================================================
# CELL 5 — Evaluation & Graph Generation
# ============================================================
print("\n🚀 Running evaluation & generating graphs...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        img = batch["img"].to(device)
        tok = model.bert.tokenizer(list(batch["q"]), padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        preds = model(img, tok["input_ids"].to(device), tok["attention_mask"].to(device)).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["ans"].numpy())

# Map indices back to answer text
idx_to_ans = {v: k for k, v in answer_vocab.items()}
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

print("\n📊 Creating Visualizations...")
os.makedirs("/kaggle/working/graphs", exist_ok=True)

# 1. Confusion Matrix (Top 15 most common classes to keep it readable)
from collections import Counter
top_classes = [item[0] for item in Counter(y_true).most_common(15)]
mask = np.isin(y_true, top_classes)
y_true_top = y_true[mask]
y_pred_top = y_pred[mask]

cm = confusion_matrix(y_true_top, y_pred_top, labels=top_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[idx_to_ans[i] for i in top_classes], yticklabels=[idx_to_ans[i] for i in top_classes])
plt.title('Confusion Matrix (Top 15 Classes)')
plt.ylabel('True Answer')
plt.xlabel('Predicted Answer')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/kaggle/working/graphs/confusion_matrix.png', dpi=300)
plt.close()

# 2. Precision & Recall per Class (Top 20 classes)
top_classes_20 = [item[0] for item in Counter(y_true).most_common(20)]
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=top_classes_20, zero_division=0)
labels_text = [idx_to_ans[i] for i in top_classes_20]

x = np.arange(len(labels_text))
width = 0.35

# Precision Plot
plt.figure(figsize=(14, 6))
plt.bar(x, precision, width, color='skyblue')
plt.ylabel('Precision')
plt.title('Precision per Class (Top 20 Most Frequent)')
plt.xticks(x, labels_text, rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/kaggle/working/graphs/precision_graph.png', dpi=300)
plt.close()

# Recall Plot
plt.figure(figsize=(14, 6))
plt.bar(x, recall, width, color='lightcoral')
plt.ylabel('Recall')
plt.title('Recall per Class (Top 20 Most Frequent)')
plt.xticks(x, labels_text, rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/kaggle/working/graphs/recall_graph.png', dpi=300)
plt.close()

print("✅ DONE! Graphs saved to /kaggle/working/graphs/")
print("   - confusion_matrix.png")
print("   - precision_graph.png")
print("   - recall_graph.png")

# ============================================================
# CELL 6 — Print & Save Test Accuracy
# ============================================================
test_accuracy = 100.0 * np.sum(y_true == y_pred) / len(y_true)

print("\n" + "="*60)
print("       VisiHealth AI — FINAL TEST RESULTS")
print("="*60)
print(f"  Total test samples : {len(y_true)}")
print(f"  Correct predictions: {int(np.sum(y_true == y_pred))}")
print(f"  ✅ TEST ACCURACY   : {test_accuracy:.4f}%")
print("="*60)

# Save results JSON
import json as _json
result = {
    "test_accuracy": round(test_accuracy, 4),
    "test_accuracy_status": "evaluated",
    "correct": int(np.sum(y_true == y_pred)),
    "total": int(len(y_true)),
    "notes": "Evaluated on SLAKE 1.0 English test set using best_checkpoint.pth"
}
Path("/kaggle/working/visihealth_test_results.json").write_text(_json.dumps(result, indent=2))
print("\n📄 Results also saved to /kaggle/working/visihealth_test_results.json")
