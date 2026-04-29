"""
VisiHealth AI - Multimodal Fusion and Answer Prediction
Phase B Upgrade: Cross-Attention Fusion + Question-Type Specific Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Multimodal Fusion — Multi-Token Version.

    The key insight: instead of squeezing both modalities to a single vector
    (length-1 sequence) before attention, we:
     - Use the BERT token sequence [B, seq_len, 768] as query (rich text context)
     - Use spatial patches from the CNN [B, num_patches, 512] as key/value

    This makes cross-attention genuinely selective: different question tokens can
    attend to different spatial regions. With length-1 sequences the attention
    weights are trivially 1.0 and the module degenerates to a linear projection.
    """
    def __init__(self, visual_dim, text_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.common_dim = 512
        # Pooled visual features: [B, visual_dim=512] → projected
        self.visual_proj = nn.Linear(visual_dim, self.common_dim)
        # Spatial features from CNN layer4: [B, P, 2048] → projected
        # (raw layer4 channels = 2048, separate from the pooled 512-dim vector)
        self.visual_spatial_proj = nn.Linear(2048, self.common_dim)
        self.text_proj   = nn.Linear(text_dim,   self.common_dim)

        # Cross-attention: Q = text tokens, K/V = visual patches
        # "Which image patches are relevant to each question token?"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True         # [B, seq, dim] format
        )

        # Self-attention on question tokens for richer text representation
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Gated fusion: combine cross-attended and self-attended text
        self.gate = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.Sigmoid()
        )

        # Layer norms for pre-norm (more stable gradient flow)
        self.norm_text   = nn.LayerNorm(self.common_dim)
        self.norm_visual = nn.LayerNorm(self.common_dim)
        self.norm_cross  = nn.LayerNorm(self.common_dim)

        # Final MLP to get fused output
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.common_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = 256

    def forward(self, visual_features, text_features, visual_seq=None, text_seq=None):
        """
        Args:
            visual_features: [B, visual_dim]         — pooled CNN feature (for residual)
            text_features:   [B, text_dim]           — BERT [CLS] embedding  (for residual)
            visual_seq:      [B, num_patches, visual_dim] — CNN spatial sequence (optional)
            text_seq:        [B, seq_len,    text_dim]    — BERT token sequence  (optional)
        Returns:
            fused: [B, 256]
        """
        # ---- Project to common dim ----
        v_global = self.visual_proj(visual_features)   # [B, 512]
        t_cls    = self.text_proj(text_features)        # [B, 512]

        # ---- Build sequences for attention ----
        # If spatial / token sequences provided, use them; otherwise fall back
        # to the pooled vectors as length-1 sequences.
        if visual_seq is not None:
            # visual_seq: [B, P, 2048]  — raw layer4 spatial channels
            v_seq = self.visual_spatial_proj(visual_seq)   # [B, P, 512]
            v_seq = self.norm_visual(v_seq)
        else:
            v_seq = v_global.unsqueeze(1)               # [B, 1, 512] — fallback

        if text_seq is not None:
            # text_seq: [B, L, text_dim]  all BERT tokens
            t_seq = self.text_proj(text_seq)            # [B, L, 512]
            t_seq = self.norm_text(t_seq)
        else:
            t_seq = t_cls.unsqueeze(1)                  # [B, 1, 512] — fallback

        # ---- Cross-attention: Q=text tokens, K/V=visual patches ----
        # Each question token attends over spatial image regions.
        cross_out, _ = self.cross_attn(
            query=t_seq,    # [B, L, 512]
            key=v_seq,      # [B, P, 512]
            value=v_seq     # [B, P, 512]
        )                   # cross_out: [B, L, 512]

        # Pool cross-attended output across question tokens → single vector
        cross_pooled = cross_out.mean(dim=1)            # [B, 512]
        cross_pooled = self.norm_cross(cross_pooled)

        # ---- Self-attention on text for richer repr ----
        self_out, _ = self.self_attn(t_seq, t_seq, t_seq)
        self_pooled = self_out.mean(dim=1)              # [B, 512]

        # ---- Gated combination ----
        gate_input = torch.cat([cross_pooled, self_pooled], dim=1)  # [B, 1024]
        gate_val   = self.gate(gate_input)                          # [B, 512]
        attended   = gate_val * cross_pooled + (1 - gate_val) * self_pooled  # [B, 512]

        # ---- Final MLP: fuse attended text with global visual ----
        fused_input = torch.cat([attended, v_global], dim=1)   # [B, 1024]
        fused = self.fusion_mlp(fused_input)                    # [B, 256]

        return fused


class QuestionTypeClassifier(nn.Module):
    """
    Classifies whether question is CLOSED (yes/no) or OPEN (specific answer).
    Used to route to the right prediction head.
    """
    def __init__(self, input_dim):
        super(QuestionTypeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)   # 0=CLOSED, 1=OPEN
        )

    def forward(self, fused_features):
        return self.classifier(fused_features)


class AnswerPredictor(nn.Module):
    """
    Dual-head answer predictor.

    CLOSED head: small set of yes/no/none/not-seen answers (fast, confident)
    OPEN   head: full vocabulary (organ names, diseases, positions, etc.)

    During training we supervise both heads.
    During inference we use the question-type classifier to pick the head.
    """
    CLOSED_ANSWERS = {
        'yes', 'no', 'none', 'not seen', 'both', 'both lungs',
        'a little', 'much', 'almost the same'
    }

    def __init__(self, input_dim, num_classes, answer_vocab=None):
        super(AnswerPredictor, self).__init__()
        self.num_classes = num_classes

        # Build closed/open index mappings from vocab
        if answer_vocab is not None:
            # answer_vocab: {answer_text: idx}
            self.closed_indices = sorted([
                idx for ans, idx in answer_vocab.items()
                if ans in self.CLOSED_ANSWERS
            ])
            self.open_indices = sorted([
                idx for ans, idx in answer_vocab.items()
                if ans not in self.CLOSED_ANSWERS
            ])
        else:
            self.closed_indices = []
            self.open_indices   = list(range(num_classes))

        num_closed = max(len(self.closed_indices), 1)
        num_open   = max(len(self.open_indices),   1)

        # CLOSED head — simple 2-layer MLP
        self.closed_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_closed)
        )

        # OPEN head — deeper to handle large vocab
        self.open_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_open)
        )

    def forward(self, fused_features):
        """
        Returns full-vocab logits assembled from both heads.
        """
        num_classes  = self.num_classes
        B            = fused_features.size(0)
        device       = fused_features.device

        # Keep full_logits in float32 — CrossEntropyLoss expects float32.
        # Cast head outputs (.to) so they always match, regardless of AMP state.
        full_logits = torch.full((B, num_classes), -1e4, device=device, dtype=torch.float32)

        # Closed head
        if self.closed_indices:
            closed_logits = self.closed_head(fused_features)          # may be float16 under AMP
            full_logits[:, self.closed_indices] = closed_logits.to(torch.float32)

        # Open head
        if self.open_indices:
            open_logits = self.open_head(fused_features)              # may be float16 under AMP
            full_logits[:, self.open_indices] = open_logits.to(torch.float32)

        return full_logits

    def get_type_logits(self, fused_features):
        """Returns separate (closed_logits, open_logits) for type-specific loss."""
        closed_logits = self.closed_head(fused_features) if self.closed_indices else None
        open_logits   = self.open_head(fused_features)   if self.open_indices   else None
        return closed_logits, open_logits


class VisiHealthModel(nn.Module):
    """
    Complete VisiHealth AI Model
    Phase B: Cross-Attention Fusion + Dual Question-Type Heads
    """
    def __init__(self, cnn_model, bert_model, config, answer_vocab=None):
        super(VisiHealthModel, self).__init__()

        self.cnn  = cnn_model
        self.bert = bert_model

        visual_dim = config['model']['cnn']['feature_dim']       # 512
        text_dim   = config['model']['bert']['hidden_size']      # 768
        num_classes = config['model']['cnn'].get('num_classes', 221)

        # Phase B: Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            num_heads=8,
            dropout=0.1
        )

        # Phase B: Dual-head answer predictor
        self.answer_predictor = AnswerPredictor(
            input_dim=self.fusion.output_dim,
            num_classes=num_classes,
            answer_vocab=answer_vocab
        )

        # Question-type classifier (CLOSED vs OPEN)
        self.qt_classifier = QuestionTypeClassifier(self.fusion.output_dim)

        # ROI Localizer (for rationale grounding)
        from models.cnn_model import ROILocalizer
        self.roi_localizer = ROILocalizer(visual_dim, num_rois=39)

    def forward(self, images, input_ids, attention_mask, return_attention=False):
        """
        Complete forward pass.
        Returns dict with answer_logits, qt_logits, roi_scores, segmentation_mask.
        """
        # --- Visual features (pooled + spatial) ---
        cnn_outputs     = self.cnn(images, return_attention=return_attention)
        visual_features = cnn_outputs['features']           # [B, 512]

        # Extract spatial feature map for multi-token cross-attention.
        # layer4 output: [B, 2048, 7, 7] → flatten spatial → [B, 49, 2048]
        # We'll project this inside the fusion module (via visual_proj).
        visual_spatial = cnn_outputs.get('spatial_features')   # [B, 49, 512] or None

        # --- Text features (CLS + all tokens) ---
        # bert() returns CLS embedding; get full token sequence separately.
        bert_all = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_all.last_hidden_state[:, 0, :]   # [B, 768]  CLS
        text_seq      = bert_all.last_hidden_state             # [B, L, 768] all tokens
        text_features = self.bert.projection(text_features)   # [B, 768] — projected CLS

        # --- Cross-Attention Fusion (multi-token) ---
        fused_features = self.fusion(
            visual_features=visual_features,
            text_features=text_features,
            visual_seq=visual_spatial,   # [B, 49, 512] if available else None → fallback
            text_seq=text_seq,           # [B, L, 768] all BERT tokens
        )  # [B, 256]

        # --- Answer prediction ---
        answer_logits = self.answer_predictor(fused_features)         # [B, num_classes]

        # --- Question-type prediction ---
        qt_logits = self.qt_classifier(fused_features)                # [B, 2]

        # --- ROI localization ---
        roi_scores = self.roi_localizer(visual_features)              # [B, 39]

        outputs = {
            'answer_logits':     answer_logits,
            'qt_logits':         qt_logits,
            'roi_scores':        roi_scores,
            'visual_features':   visual_features,
            'text_features':     text_features,
            'fused_features':    fused_features,
            'segmentation_mask': cnn_outputs['segmentation_mask'],
        }

        if return_attention:
            outputs['attention_maps'] = cnn_outputs.get('attention_maps', None)

        return outputs

    def predict(self, images, questions, device='cuda', image_size: int = 224):
        """
        High-level prediction interface.

        Args:
            images:     A PIL Image, list of PIL Images, or pre-processed tensor [B,3,H,W]
            questions:  A string or list of question strings
            device:     'cuda' or 'cpu'
            image_size: Resize target — must match what the model was trained at (default 224).
                        Read from config['image']['size'] when calling from test_model.py.
        """
        self.eval()
        with torch.no_grad():
            # Tokenize text
            if isinstance(questions, str):
                questions = [questions]
            tokens         = self.bert.tokenize(questions)
            input_ids      = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            # Preprocess images if raw PIL images were passed
            if not isinstance(images, torch.Tensor):
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                if not isinstance(images, (list, tuple)):
                    images = [images]
                images = torch.stack([transform(img) for img in images])

            images = images.to(device)

            # forward() handles spatial CNN features + full BERT token sequence
            # internally — multi-token cross-attention is automatic.
            outputs = self.forward(images, input_ids, attention_mask, return_attention=True)

            answer_probs      = torch.softmax(outputs['answer_logits'], dim=1)
            predicted_answers = torch.argmax(answer_probs, dim=1)
            top_rois          = torch.topk(outputs['roi_scores'], k=3, dim=1)

            return {
                'predicted_answers': predicted_answers.cpu(),
                'answer_probs':      answer_probs.cpu(),
                'top_rois':          top_rois.indices.cpu(),
                'roi_scores':        top_rois.values.cpu(),
                'attention_maps':    outputs.get('attention_maps'),
                'segmentation_mask': outputs['segmentation_mask'].cpu(),
            }



def build_visihealth_model(config, cnn_model, bert_model, answer_vocab=None):
    """Factory function to build the complete model."""
    return VisiHealthModel(cnn_model, bert_model, config, answer_vocab=answer_vocab)


if __name__ == "__main__":
    import yaml
    from models.cnn_model import get_cnn_model
    from models.bert_model import get_bert_model

    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    cnn   = get_cnn_model(config)
    bert  = get_bert_model(config)
    model = build_visihealth_model(config, cnn, bert)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Test forward pass — use actual image_size from config (336)
    img_size   = config['image']['size']
    images     = torch.randn(2, 3, img_size, img_size)
    input_ids  = torch.randint(0, 1000, (2, 128))
    attn_mask  = torch.ones(2, 128, dtype=torch.long)
    outputs    = model(images, input_ids, attn_mask)

    print(f"answer_logits shape: {outputs['answer_logits'].shape}")
    print(f"qt_logits shape:     {outputs['qt_logits'].shape}")
    print(f"roi_scores shape:    {outputs['roi_scores'].shape}")
    print("--- Forward pass OK!")

