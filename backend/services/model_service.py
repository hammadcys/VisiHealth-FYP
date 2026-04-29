"""
VisiHealth AI - Model Service
Handles model loading, caching, and inference
"""

import torch
import yaml
import json
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import get_cnn_model, get_bert_model, build_visihealth_model
from utils.knowledge_graph import load_knowledge_graph, RationaleGenerator

# Use the SAME ROI name mapping that was used during training (from RationaleGenerator).
# This is the authoritative source — do NOT define a separate mapping here.
# 0=liver, 1=kidney, 2=spleen, 3=pancreas, 4=gallbladder, 5=stomach,
# 6=intestine, 7=lung, 8=heart, 9=aorta, 10=brain, ...
ROI_REGION_NAMES = RationaleGenerator.ROI_NAMES


class ModelService:
    """Singleton service for model management"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model service (only once)"""
        if ModelService._initialized:
            return
        
        self.model = None
        self.device = None
        self.config = None
        self.answer_vocab = None
        self.idx_to_answer = None
        self.kg = None
        self.rationale_gen = None
        self.transform = None
        self.max_question_length = 128
        
        ModelService._initialized = True
    
    def load_model(self, config_path, checkpoint_path, model_info_path, kg_file_path):
        """
        Load model, checkpoint, and knowledge graph
        
        Args:
            config_path: Path to config.yaml
            checkpoint_path: Path to model checkpoint
            model_info_path: Path to model info JSON
            kg_file_path: Path to knowledge graph file
        """
        print("--- Loading VisiHealth AI Model (Latest - Cross-Attention Fusion)...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self.device}")
        
        # Load answer vocabulary from model info
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Convert string keys to integers for answer vocab
        self.answer_vocab = {v: int(k) for k, v in model_info['answer_vocab'].items()}
        self.idx_to_answer = {int(k): v for k, v in model_info['answer_vocab'].items()}
        num_classes = len(self.answer_vocab)
        self.image_size = int(model_info.get('image_size', self.config.get('image', {}).get('size', 336)))
        
        print(f"   Answer classes: {num_classes}")
        print(f"   Image size: {self.image_size}x{self.image_size}")
        
        # Update config with correct number of classes from the saved vocab
        self.config['model']['cnn']['num_classes'] = num_classes
        if 'fusion' not in self.config:
            self.config['fusion'] = {}
        self.config['fusion']['num_classes'] = num_classes
        
        # Build model architecture
        cnn = get_cnn_model(self.config).to(self.device)
        bert = get_bert_model(self.config).to(self.device)
        self.model = build_visihealth_model(
            self.config,
            cnn,
            bert,
            answer_vocab=self.answer_vocab
        ).to(self.device)
        
        # Load checkpoint weights
        print(f"   Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        epoch = checkpoint.get('epoch', 'N/A')
        val_acc = checkpoint.get('best_val_acc', 0.0)
        val_acc_str = f"{val_acc:.2f}%" if isinstance(val_acc, (int, float)) else str(val_acc)
        print(f"   --- Model loaded (Epoch: {epoch}, Val Acc: {val_acc_str})")
        
        # Load knowledge graph
        if kg_file_path and Path(kg_file_path).exists():
            print(f"   Loading knowledge graph...")
            self.kg = load_knowledge_graph(str(kg_file_path))
            self.rationale_gen = RationaleGenerator(self.kg)
            print(f"   --- Knowledge graph loaded")
        else:
            print(f"   !!! KG file not found at {kg_file_path} - rationale disabled")
        
        # Setup image transform — use image_size from model_info (matches training exactly)
        self.max_question_length = int(
            self.config.get('model', {}).get('bert', {}).get('max_length', 128)
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"--- VisiHealth AI Ready! (fusion=cross_attention, img={self.image_size}px, classes={num_classes})")
    
    def predict(self, image, question):
        """
        Make prediction on image and question
        
        Args:
            image: PIL Image or file path
            question: Question string
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or file path")
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Tokenize question
        tokens = self.model.bert.tokenize([question], max_length=self.max_question_length)
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor, input_ids, attention_mask, return_attention=True)
        
        # Get predictions
        answer_logits = outputs['answer_logits']
        answer_probs = torch.softmax(answer_logits, dim=1)[0]
        
        # Top 3 predictions
        top_probs, top_indices = torch.topk(answer_probs, k=min(3, len(answer_probs)))

        predicted_idx = top_indices[0].item()
        predicted_answer = self.idx_to_answer.get(predicted_idx, 'unknown')
        confidence = top_probs[0].item()
        
        # ROI scores — ROILocalizer was never supervised so argmax is meaningless.
        # Instead, derive the displayed region from the predicted answer text.
        roi_scores = outputs['roi_scores'][0].cpu()

        # Build a reverse lookup: organ name → ROI index
        _name_to_idx = {v: k for k, v in ROI_REGION_NAMES.items()}

        # Medical condition → associated organ (for when the answer is a disease/condition)
        _condition_to_organ = {
            'cardiomegaly': 'heart',      'cardiomegalia': 'heart',
            'pneumonia': 'lung',          'pneumothorax': 'lung',
            'atelectasis': 'lung',        'effusion': 'lung',
            'edema': 'lung',              'emphysema': 'lung',
            'fibrosis': 'lung',           'consolidation': 'lung',
            'hepatomegaly': 'liver',      'cirrhosis': 'liver',
            'hepatitis': 'liver',         'fatty liver': 'liver',
            'splenomegaly': 'spleen',     'nephritis': 'kidney',
            'hydronephrosis': 'kidney',   'nephrolithiasis': 'kidney',
            'pancreatitis': 'pancreas',   'cholecystitis': 'gallbladder',
            'cholelithiasis': 'gallbladder',
            'stroke': 'brain',            'glioma': 'brain',
            'meningioma': 'brain',        'hydrocephalus': 'brain',
            'alzheimer': 'brain',         'aneurysm': 'aorta',
            'atherosclerosis': 'vessel',  'thrombosis': 'vessel',
            'fracture': 'bone',           'osteoporosis': 'bone',
            'scoliosis': 'spine',         'herniation': 'spine',
        }
        
        # Broader body regions that aren't in SLAKE's 39 specific organs
        _custom_regions = {'abdomen', 'chest', 'pelvis', 'head', 'neck', 'thorax', 'breast'}

        # Check if the predicted answer itself is a known region name
        answer_lower = predicted_answer.lower().strip()
        if answer_lower in _name_to_idx:
            # Answer IS an organ/region — use it directly
            display_roi_idx  = _name_to_idx[answer_lower]
            display_roi_name = answer_lower
            display_roi_conf = float(confidence)
        elif answer_lower in _custom_regions:
            # Answer is a broad body region
            display_roi_idx  = -1
            display_roi_name = answer_lower
            display_roi_conf = float(confidence)
        elif answer_lower in _condition_to_organ:
            # Answer is a condition — map to its associated organ
            organ = _condition_to_organ[answer_lower]
            display_roi_idx  = _name_to_idx.get(organ, -1)
            display_roi_name = organ
            display_roi_conf = float(confidence)
        else:
            # Check if any organ name appears as a substring in the answer
            matched = next(
                ((name, idx) for name, idx in _name_to_idx.items() if name in answer_lower),
                None
            )
            # Also check condition map for partial match
            if not matched:
                matched_cond = next(
                    (cond for cond in _condition_to_organ if cond in answer_lower),
                    None
                )
                if matched_cond:
                    organ = _condition_to_organ[matched_cond]
                    matched = (organ, _name_to_idx.get(organ, -1))

            if matched:
                display_roi_name, display_roi_idx = matched
                display_roi_conf = float(confidence)
            else:
                # No match — show N/A honestly
                display_roi_idx  = -1
                display_roi_name = 'N/A'
                display_roi_conf = 0.0
        
        # Generate rationale if KG available
        rationale = ""
        if self.rationale_gen:
            rationale = self.rationale_gen.generate_rationale(
                predicted_answer=predicted_answer,
                confidence=confidence,
                top_roi_indices=[display_roi_idx] if display_roi_idx >= 0 else [],
                roi_scores=[display_roi_conf] if display_roi_idx >= 0 else [0.0],
                question=question
            )
        
        # Build response
        result = {
            'answer': predicted_answer,
            'confidence': float(confidence),
            'top_predictions': [
                {
                    'answer': self.idx_to_answer.get(idx.item(), 'unknown'),
                    'confidence': float(prob.item())
                }
                for idx, prob in zip(top_indices, top_probs)
            ],
            'roi': {
                'top_region': display_roi_idx,
                'region_name': display_roi_name,
                'confidence': display_roi_conf
            },
            'rationale': rationale,
            'attention_available': 'attention_maps' in outputs
        }
        
        # Add attention map if requested
        if 'attention_maps' in outputs and outputs['attention_maps']:
            # Get layer4 attention map (most detailed)
            attention = outputs['attention_maps'].get('layer4')
            if attention is not None:
                # Convert to numpy and resize
                attn_map = attention[0, 0].cpu().numpy()
                # Normalize to 0-1
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                result['attention_map'] = attn_map.tolist()
        
        return result
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
        
        return {
            'device': str(self.device),
            'num_classes': len(self.answer_vocab),
            'image_size': getattr(self, 'image_size', 336),
            'fusion_method': 'cross_attention',
            'kg_enabled': self.kg is not None,
            'model_loaded': True,
            'model_version': 'Latest (Phase A+B+C + Bug Fixes)'
        }


# Global model service instance
model_service = ModelService()
