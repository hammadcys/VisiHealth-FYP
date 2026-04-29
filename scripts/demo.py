"""
VisiHealth AI - Demo Script
Test the complete VisiHealth AI system with sample inputs
"""

import os
import sys
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import get_cnn_model, get_bert_model, build_visihealth_model
from data import SLAKEDataset
from utils.knowledge_graph import load_knowledge_graph, RationaleGenerator


def visualize_attention(image, attention_map, save_path=None):
    """Visualize attention map overlaid on image"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = np.array(image) / 255.0
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention overlay
    if attention_map is not None:
        if isinstance(attention_map, torch.Tensor):
            attention_np = attention_map.cpu().squeeze().numpy()
        else:
            attention_np = attention_map
        
        axes[1].imshow(image_np)
        axes[1].imshow(attention_np, alpha=0.5, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def demo_inference(config_path='../config.yaml', checkpoint_path=None):
    """
    Run demo inference with VisiHealth AI
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to trained model checkpoint
    """
    # Load config
    print("="*70)
    print(" "*20 + "VisiHealth AI Demo")
    print("="*70)
    
    # Handle relative paths from scripts directory
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load models
    print("\nLoading models...")
    cnn = get_cnn_model(config)
    bert = get_bert_model(config)
    model = build_visihealth_model(config, cnn, bert)
    
    # Load trained weights if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("Warning: No checkpoint loaded. Using randomly initialized weights.")
        print("For actual inference, please provide a trained checkpoint.")
    
    model = model.to(device)
    model.eval()
    
    # Load knowledge graph
    kg_file = config['dataset'].get('kg_file', './data/SLAKE/knowledge_graph.txt')
    if os.path.exists(kg_file):
        print(f"\nLoading knowledge graph from {kg_file}")
        kg = load_knowledge_graph(kg_file)
        rationale_gen = RationaleGenerator(kg)
    else:
        print(f"Warning: KG file not found at {kg_file}")
        kg = None
        rationale_gen = None
    
    # Load a sample from dataset or use custom input
    print("\n" + "="*70)
    print("Running Sample Inference...")
    print("="*70)
    
    data_dir = config['dataset']['root_dir']
    
    # Try to load from dataset
    try:
        dataset = SLAKEDataset(
            data_dir=data_dir,
            split='validate',
            tokenizer_name=config['model']['bert']['model_name'],
            augment=False
        )
        
        # Get a random sample
        sample_idx = np.random.randint(0, len(dataset))
        sample = dataset[sample_idx]
        
        print(f"\nSample {sample_idx} from validation set:")
        print(f"  Question: {sample['question_text']}")
        print(f"  Ground Truth Answer: {sample['answer_text']}")
        print(f"  Image: {sample['img_name']}")
        
        # Prepare input
        image = sample['image'].unsqueeze(0).to(device)
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(image, input_ids, attention_mask, return_attention=True)
        
        # Get predictions
        answer_logits = outputs['answer_logits']
        answer_probs = torch.softmax(answer_logits, dim=1)
        predicted_answer_idx = torch.argmax(answer_probs, dim=1).item()
        confidence = answer_probs[0, predicted_answer_idx].item()
        
        predicted_answer_text = dataset.get_answer_text(predicted_answer_idx)
        
        # Get ROI information
        roi_scores = outputs['roi_scores'][0]
        top_k_rois = torch.topk(roi_scores, k=3)
        
        print("\n" + "-"*70)
        print("PREDICTION RESULTS:")
        print("-"*70)
        print(f"  Predicted Answer: {predicted_answer_text}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"  Correct: {'✓' if predicted_answer_text == sample['answer_text'] else '✗'}")
        
        print(f"\n  Top 3 Detected ROIs:")
        for i, (score, idx) in enumerate(zip(top_k_rois.values, top_k_rois.indices)):
            roi_name = rationale_gen.ROI_NAMES.get(idx.item(), f'ROI_{idx.item()}') if rationale_gen else f'ROI_{idx.item()}'
            print(f"    {i+1}. {roi_name}: {score.item():.4f}")
        
        # Generate rationale
        if rationale_gen:
            roi_names = [rationale_gen.ROI_NAMES.get(idx.item(), f'ROI_{idx.item()}') 
                        for idx in top_k_rois.indices]
            
            rationale = rationale_gen.generate_rationale(
                predicted_answer=predicted_answer_text,
                confidence=confidence,
                top_roi_indices=top_k_rois.indices.tolist(),
                roi_scores=top_k_rois.values.tolist(),
                question=sample['question_text']
            )
            
            print(f"\n  Rationale:")
            print(f"    {rationale}")
        
        # Visualize attention if available
        if 'attention_maps' in outputs and outputs['attention_maps'] is not None:
            attention_map = outputs['attention_maps']['layer4'][0, 0]  # First batch, first channel
            
            # Resize attention map to image size
            import torch.nn.functional as F
            attention_map_resized = F.interpolate(
                attention_map.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )[0, 0]
            
            print(f"\n  Generating attention visualization...")
            visualize_attention(
                sample['image'],
                attention_map_resized,
                save_path='./attention_visualization.png'
            )
        
        print("\n" + "="*70)
        print("Demo completed successfully!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\nError: Dataset not found - {e}")
        print("\nTo run this demo, you need:")
        print("  1. Download the SLAKE dataset from: https://www.med-vqa.com/slake/")
        print("  2. Place it in: ./data/SLAKE/")
        print("  3. Ensure the structure:")
        print("     ./data/SLAKE/")
        print("       ├── imgs/")
        print("       ├── train.json")
        print("       ├── validate.json")
        print("       └── test.json")
        print("\nFor now, here's what the system would do:")
        print("  1. Load medical image and question")
        print("  2. Extract visual features with custom CNN")
        print("  3. Encode question with BioBERT")
        print("  4. Fuse multimodal features")
        print("  5. Predict answer with confidence")
        print("  6. Localize relevant ROIs")
        print("  7. Retrieve KG facts")
        print("  8. Generate human-readable rationale")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VisiHealth AI Demo')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_checkpoint.pth',
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    demo_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
