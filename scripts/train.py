"""
VisiHealth AI - Training Script
Complete training pipeline with multi-task learning, early stopping, and evaluation
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import get_cnn_model, get_bert_model, build_visihealth_model
from data import get_dataloader
from utils.knowledge_graph import load_knowledge_graph, RationaleGenerator
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for severe class imbalance.
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    gamma > 0 down-weights easy (well-classified) examples so the model
    focuses on hard / rare ones. gamma=2 is standard. Pair with class weights
    (alpha) from compute_class_weights for best results.

    Compatible with label smoothing via soft targets.
    """
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight          # per-class alpha weights  [C]
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, C]  raw (unnormalized) scores
            targets: [B]     integer class indices
        Returns:
            Scalar loss.
        """
        C = logits.size(1)
        B = logits.size(0)

        # --- Soft target for label smoothing ---
        if self.label_smoothing > 0:
            with torch.no_grad():
                soft = torch.full_like(logits, self.label_smoothing / (C - 1))
                soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            soft = None

        # --- Log-softmax ---
        log_p = F.log_softmax(logits, dim=1)          # [B, C]
        p     = log_p.exp()                           # [B, C]

        # Probability of the correct class (for focal weight)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

        # --- Cross-entropy component ---
        if soft is not None:
            # Soft cross-entropy: -sum_c soft[c] * log_p[c]
            ce = -(soft * log_p).sum(dim=1)            # [B]
        else:
            ce = F.nll_loss(log_p, targets, reduction='none')  # [B]

        # --- Focal weight ---
        focal_weight = (1.0 - p_t) ** self.gamma      # [B]

        loss = focal_weight * ce                       # [B]

        # --- Per-class alpha weighting ---
        if self.weight is not None:
            alpha_t = self.weight[targets]             # [B]
            loss = alpha_t * loss

        return loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """VisiHealth AI Trainer"""
    
    def __init__(self, config_path='../config.yaml'):
        """Initialize trainer with configuration"""
        # Handle relative paths from scripts directory
        if not os.path.isabs(config_path):
            # Try relative to script directory first
            script_relative = os.path.join(os.path.dirname(__file__), config_path)
            # Also try current working directory
            cwd_path = os.path.join(os.getcwd(), config_path.replace('../', ''))
            
            if os.path.exists(script_relative):
                config_path = script_relative
            elif os.path.exists(cwd_path):
                config_path = cwd_path
            elif os.path.exists(config_path.replace('../', '')):
                config_path = config_path.replace('../', '')
            else:
                # Last resort: try root project directory
                config_path = os.path.join(os.getcwd(), 'config.yaml')
        
        # Load config
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(
            self.config['system']['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        self.seed = self.config['system']['seed']
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Create directories
        self.save_dir = self.config['system']['save_dir']
        self.log_dir = self.config['system']['log_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tensorboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.log_dir, f'run_{timestamp}'))
        
        # Load datasets
        print("\n" + "="*60)
        print("Loading Datasets...")
        print("="*60)
        image_size = self.config['image']['size']   # 336
        self.train_loader, self.train_dataset = get_dataloader(
            data_dir=self.config['dataset']['root_dir'],
            split='train',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['system']['num_workers'],
            tokenizer_name=self.config['model']['bert']['model_name'],
            image_size=image_size
        )

        # WeightedRandomSampler — oversample CLOSED (yes/no) questions 2x
        # to reduce the CLOSED vs OPEN accuracy gap
        from torch.utils.data import WeightedRandomSampler, DataLoader as TorchDataLoader
        _closed_answers = {
            'yes', 'no', 'none', 'not seen', 'both', 'both lungs',
            'a little', 'much', 'almost the same'
        }
        _sample_weights = []
        for item in self.train_dataset.data:
            ans = self.train_dataset._normalize_answer(item['answer'])
            _sample_weights.append(2.0 if ans in _closed_answers else 1.0)
        _sampler = WeightedRandomSampler(
            weights=_sample_weights,
            num_samples=len(_sample_weights),
            replacement=True
        )
        self.train_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=_sampler,          # sampler replaces shuffle=True
            num_workers=self.config['system']['num_workers'],
            pin_memory=True
        )
        print("✅ WeightedRandomSampler: CLOSED questions oversampled 2x")

        # Use training vocabulary for validation to ensure consistency
        self.val_loader, self.val_dataset = get_dataloader(
            data_dir=self.config['dataset']['root_dir'],
            split='validate',
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['system']['num_workers'],
            tokenizer_name=self.config['model']['bert']['model_name'],
            train_vocab=self.train_dataset.answer_vocab,  # ← Share vocabulary
            image_size=image_size
        )
        
        # Update num_classes in config
        self.num_classes = self.train_dataset.num_classes
        self.config['model']['cnn']['num_classes'] = self.num_classes
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Answer vocabulary size: {self.num_classes}")
        
        # Verify vocabulary consistency
        assert self.train_dataset.num_classes == self.val_dataset.num_classes, \
            f"Vocabulary mismatch! Train: {self.train_dataset.num_classes}, Val: {self.val_dataset.num_classes}"
        print("✅ Train and validation vocabularies match!")
        
        # Compute class weights for imbalanced SLAKE dataset (after num_classes is set)
        self.class_weights = self._compute_class_weights()
        
        # Build model
        print("\n" + "="*60)
        print("Building Model...")
        print("="*60)
        self.cnn = get_cnn_model(self.config).to(self.device)
        self.bert = get_bert_model(self.config).to(self.device)
        # Phase B: Pass answer_vocab so dual-head predictor can build closed/open splits
        self.model = build_visihealth_model(
            self.config, self.cnn, self.bert,
            answer_vocab=self.train_dataset.answer_vocab
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss functions
        label_smoothing = self.config['training'].get('label_smoothing', 0.1)
        use_focal = self.config['training'].get('use_focal_loss', False)
        focal_gamma = self.config['training'].get('focal_gamma', 2.0)

        if use_focal:
            print(f"✅ Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
            self.vqa_criterion = FocalLoss(
                gamma=focal_gamma,
                weight=self.class_weights,
                label_smoothing=label_smoothing
            )
        else:
            # Phase C: Label smoothing prevents overconfident predictions
            print(f"✅ Using CrossEntropyLoss (label_smoothing={label_smoothing}, class_weights=True)")
            self.vqa_criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=label_smoothing
            )
        self.seg_criterion = nn.BCEWithLogitsLoss()  # Safe for mixed precision
        # Question-type auxiliary loss (CLOSED=0, OPEN=1)
        self.qt_criterion = nn.CrossEntropyLoss()
        
        # Optimizer - Use AdamW for better weight decay handling
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training']['learning_rate']
        
        # Define parameter groups for differential learning rates
        bert_params = list(self.bert.parameters())
        cnn_params = list(self.cnn.parameters())
        fusion_params = [p for p in self.model.parameters() 
                        if id(p) not in [id(bp) for bp in bert_params] 
                        and id(p) not in [id(cp) for cp in cnn_params]]
        
        # Store base learning rates for warmup
        self.base_lrs = [lr * 0.1, lr, lr]  # BERT, CNN, Fusion
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': bert_params, 'lr': lr * 0.1},   # 10x smaller for pretrained BERT
                {'params': cnn_params, 'lr': lr},          # Base LR for pretrained ResNet50
                {'params': fusion_params, 'lr': lr}        # Base LR for fusion layers
            ], weight_decay=self.config['training']['weight_decay'])
        else:  # adam
            self.optimizer = optim.Adam([
                {'params': bert_params, 'lr': 2e-5},
                {'params': cnn_params, 'lr': 3e-4},
                {'params': fusion_params, 'lr': 3e-4}
            ], weight_decay=self.config['training']['weight_decay'])
        
        # Scheduler - Support both cosine annealing and reduce on plateau
        scheduler_name = self.config['training'].get('scheduler', 'reduce_on_plateau').lower()
        
        if scheduler_name == 'cosine_annealing':
            # Cosine annealing with warmup
            warmup_epochs = self.config['training'].get('warmup_epochs', 10)
            total_epochs = self.config['training']['num_epochs']
            min_lr = self.config['training'].get('min_lr', 1e-6)
            
            # Main cosine scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr
            )
            self.warmup_epochs = warmup_epochs
            self.use_warmup = True
        else:  # reduce_on_plateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['training']['patience']
            )
            self.use_warmup = False
        
        # Early stopping
        if self.config['training']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.config['training']['early_stopping']['patience'],
                min_delta=self.config['training']['early_stopping']['min_delta']
            )
        else:
            self.early_stopping = None
        
        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # Load knowledge graph (for evaluation)
        kg_file = self.config['dataset'].get('kg_file', './data/SLAKE/knowledge_graph.txt')
        if os.path.exists(kg_file):
            self.kg = load_knowledge_graph(kg_file)
            self.rationale_gen = RationaleGenerator(self.kg)
        else:
            print(f"Warning: KG file not found at {kg_file}")
            self.kg = None
            self.rationale_gen = None
    
    def _compute_class_weights(self):
        """Compute class weights for handling imbalanced SLAKE dataset"""
        from collections import Counter
        
        # Count answer distribution using NORMALIZED answers to match vocab
        all_answers = []
        for item in self.train_dataset.data:
            # Phase A: Must normalize here to match the vocab built with normalization
            answer_text = self.train_dataset._normalize_answer(item['answer'])
            answer_idx = self.train_dataset.answer_vocab.get(answer_text, 0)
            all_answers.append(answer_idx)
        
        answer_counts = Counter(all_answers)
        total_samples = len(all_answers)
        
        # Compute weights: capped inverse frequency (cap at 10x to avoid extreme values)
        weights = torch.zeros(self.num_classes)
        for class_idx in range(self.num_classes):
            count = answer_counts.get(class_idx, 1)  # Avoid division by zero
            raw_weight = total_samples / (self.num_classes * count)
            weights[class_idx] = min(raw_weight, 10.0)  # Cap at 10x
        
        print(f"✅ Computed class weights for {self.num_classes} classes (capped at 10x)")
        return weights.to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_vqa_loss = 0.0
        total_seg_loss = 0.0
        correct = 0
        total = 0
        
        # Gradient accumulation setup
        accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # Mixed precision setup
        use_amp = self.config['training'].get('mixed_precision', False)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answers = batch['answer'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = self.model(images, input_ids, attention_mask)
                
                # VQA loss
                vqa_loss = self.vqa_criterion(outputs['answer_logits'], answers)
                
                # Question-type auxiliary loss (Phase B)
                # Build qt_targets: CLOSED=0 if answer in closed set, OPEN=1 otherwise
                qt_loss = torch.tensor(0.0, device=self.device)
                if 'qt_logits' in outputs and outputs['qt_logits'] is not None:
                    closed_idx_set = set(self.model.answer_predictor.closed_indices)
                    qt_targets = torch.tensor(
                        [0 if a.item() in closed_idx_set else 1 for a in answers],
                        dtype=torch.long, device=self.device
                    )
                    qt_loss = self.qt_criterion(outputs['qt_logits'], qt_targets)
                
                # Segmentation loss (multi-task learning)
                if self.config['training']['multitask']['enabled']:
                    seg_preds = outputs['segmentation_mask'].squeeze(1)
                    
                    # Upsample predictions to match mask size
                    seg_preds = torch.nn.functional.interpolate(
                        seg_preds.unsqueeze(1), 
                        size=masks.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(1)
                    
                    seg_loss = self.seg_criterion(seg_preds, masks)
                    
                    vqa_weight = self.config['training']['multitask']['vqa_weight']
                    seg_weight = self.config['training']['multitask']['segmentation_weight']
                    loss = vqa_weight * vqa_loss + seg_weight * seg_loss + 0.1 * qt_loss
                    
                    total_seg_loss += seg_loss.item()
                else:
                    loss = vqa_loss + 0.1 * qt_loss
                
                # Scale loss for gradient accumulation
                loss = loss / accum_steps
            
            # Backward pass with gradient scaling
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                if use_amp:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * accum_steps  # Unscale for logging
            total_vqa_loss += vqa_loss.item()
            
            _, predicted = outputs['answer_logits'].max(1)
            total += answers.size(0)
            correct += predicted.eq(answers).sum().item()
            
            # Update progress bar with current batch stats
            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f'{current_avg_loss:.4f}', acc=f'{100.*correct/total:.2f}%')
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item() * accum_steps, global_step)
            

        
        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        avg_vqa_loss = total_vqa_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader) if self.config['training']['multitask']['enabled'] else 0
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'vqa_loss': avg_vqa_loss,
            'seg_loss': avg_seg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate the model with per-type breakdown (CLOSED vs OPEN)"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Track CLOSED vs OPEN accuracy separately
        closed_idx_set = set(self.model.answer_predictor.closed_indices)
        closed_correct = 0
        closed_total   = 0
        open_correct   = 0
        open_total     = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answers = batch['answer'].to(self.device)

            # Forward pass
            outputs = self.model(images, input_ids, attention_mask)

            # Loss (use CrossEntropyLoss for val even if focal for train — comparable numbers)
            loss = F.cross_entropy(
                outputs['answer_logits'].float(),
                answers,
                weight=self.class_weights
            )
            total_loss += loss.item()

            # Overall accuracy
            _, predicted = outputs['answer_logits'].max(1)
            total += answers.size(0)
            correct += predicted.eq(answers).sum().item()

            # CLOSED vs OPEN breakdown
            for pred, gt in zip(predicted, answers):
                gt_item = gt.item()
                if gt_item in closed_idx_set:
                    closed_total += 1
                    if pred.item() == gt_item:
                        closed_correct += 1
                else:
                    open_total += 1
                    if pred.item() == gt_item:
                        open_correct += 1

            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f'{current_avg_loss:.4f}', acc=f'{100.*correct/total:.2f}%')

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        closed_acc = 100. * closed_correct / closed_total if closed_total > 0 else 0.0
        open_acc   = 100. * open_correct   / open_total   if open_total   > 0 else 0.0

        print(f"  [Val Breakdown] CLOSED: {closed_acc:.2f}% ({closed_correct}/{closed_total}) | "
              f"OPEN: {open_acc:.2f}% ({open_correct}/{open_total})")

        return {
            'loss':       avg_loss,
            'accuracy':   accuracy,
            'closed_acc': closed_acc,
            'open_acc':   open_acc,
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping.counter if self.early_stopping else 0,
            'early_stopping_best_loss': self.early_stopping.best_loss if self.early_stopping else None,
            'config': self.config
        }
        
        # Save last checkpoint
        path = os.path.join(self.save_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, path)
        
        # Save periodic checkpoints every 20 epochs (not 5 - saves space!)
        # Keep only the 3 most recent periodic checkpoints
        if (self.current_epoch + 1) % 20 == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.current_epoch + 1}.pth')
            torch.save(checkpoint, path)
            print(f"Saved checkpoint at epoch {self.current_epoch + 1}")
            
            # Clean up old periodic checkpoints (keep only 3 most recent)
            periodic_checkpoints = sorted([
                f for f in os.listdir(self.save_dir) 
                if f.startswith('checkpoint_epoch_') and f.endswith('.pth')
            ])
            if len(periodic_checkpoints) > 3:
                for old_ckpt in periodic_checkpoints[:-3]:
                    old_path = os.path.join(self.save_dir, old_ckpt)
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {old_ckpt}")
        
        # Save best checkpoint
        if is_best:
            path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, path)
            print(f"Saved best model (acc: {self.best_val_acc:.2f}%)")
    
    def save_model_info(self):
        """
        Save VisiHealth_Model_Info.json to results/ after training completes.

        This file is used by:
          - test_model.py       (loads answer vocab for inference)
          - test_load_model.py  (reconstructs model classes)
          - The web app backend (maps class indices → answer text)

        The vocab saved here is ALWAYS the normalized vocab built from the
        training data, so it matches the checkpoint exactly.
        """
        os.makedirs(self.config['system']['results_dir'], exist_ok=True)
        info_path = os.path.join(
            self.config['system']['results_dir'],
            'VisiHealth_Model_Info.json'
        )

        # Invert vocab: {answer_text: idx} → {idx: answer_text}  (JSON needs str keys)
        idx_to_answer = {str(idx): ans for ans, idx in self.train_dataset.answer_vocab.items()}

        model_info = {
            'model_type':      'VisiHealth AI',
            'checkpoint_path': os.path.join(self.save_dir, 'best_checkpoint.pth'),
            'test_accuracy':   None,          # filled in after evaluation
            'num_classes':     self.num_classes,
            'answer_vocab':    idx_to_answer,  # the clean normalized vocab
            'image_size':      self.config['image']['size'],
            'bert_model':      self.config['model']['bert']['model_name'],
            'usage': {
                'input':  f"Medical image ({self.config['image']['size']}x{self.config['image']['size']}) + question text",
                'output': 'Answer + confidence + ROI scores + rationale'
            },
            'training_info': {
                'dataset':       'SLAKE 1.0',
                'total_samples': len(self.train_dataset),
                'epochs':        self.current_epoch + 1,
                'best_val_acc':  self.best_val_acc,
                'platform':      'Local / Kaggle'
            }
        }

        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Model info saved to: {info_path}")
        print(f"   Classes: {self.num_classes} | Best val acc: {self.best_val_acc:.2f}%")
        print(f"   Vocab is clean normalized — matches the trained checkpoint.")

    def load_checkpoint(self, resume_path=None):
        """Load checkpoint to resume training"""
        if resume_path and os.path.exists(resume_path):
            checkpoint_path = resume_path
        else:
            # Find latest checkpoint
            checkpoints = sorted([f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_epoch_')])
            if not checkpoints:
                print("No checkpoint found. Starting from scratch.")
                return False
            checkpoint_path = os.path.join(self.save_dir, checkpoints[-1])
        
        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {checkpoint_path}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Restore early stopping state if available
        if self.early_stopping and 'early_stopping_counter' in checkpoint:
            self.early_stopping.counter = checkpoint['early_stopping_counter']
            self.early_stopping.best_loss = checkpoint['early_stopping_best_loss']
        
        print(f"✅ Resumed from epoch {self.start_epoch}")
        print(f"   Best val accuracy so far: {self.best_val_acc:.2f}%")
        print(f"   Best val loss so far: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return True
    
    def train(self, resume=False, resume_path=None):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint(resume_path)
        
        print("\n" + "="*60)
        print("Starting Training...")
        print("="*60)
        
        for epoch in range(self.start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate warmup
            if self.use_warmup and epoch < self.warmup_epochs:
                # Linear warmup - set LR to base_lr * warmup_factor
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.base_lrs[i] * warmup_factor
            
            # Update scheduler
            if hasattr(self.scheduler, 'step'):
                # For ReduceLROnPlateau, pass metrics
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                # For CosineAnnealing, just step
                elif self.use_warmup and epoch >= self.warmup_epochs:
                    self.scheduler.step()
                elif not self.use_warmup:
                    self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")

            self.writer.add_scalar('Train/Loss',        train_metrics['loss'],       epoch)
            self.writer.add_scalar('Train/Accuracy',    train_metrics['accuracy'],   epoch)
            self.writer.add_scalar('Val/Loss',          val_metrics['loss'],         epoch)
            self.writer.add_scalar('Val/Accuracy',      val_metrics['accuracy'],     epoch)
            # Log per-type accuracy to TensorBoard for diagnostics
            self.writer.add_scalar('Val/CLOSED_Accuracy', val_metrics.get('closed_acc', 0), epoch)
            self.writer.add_scalar('Val/OPEN_Accuracy',   val_metrics.get('open_acc',   0), epoch)
            
            # Save best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_metrics['loss'])
                if self.early_stopping.early_stop:
                    print("\nEarly stopping triggered!")
                    break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*60)

        # Auto-save model info JSON — used by test scripts and web app
        self.save_model_info()

        self.writer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VisiHealth AI')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to resume from')
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = Trainer(config_path=args.config)
    trainer.train(resume=args.resume, resume_path=args.checkpoint)
