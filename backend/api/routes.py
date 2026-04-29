"""
VisiHealth AI - API Routes
"""

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import io
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from backend.services.model_service import model_service
from backend.utils.validators import validate_image_file, validate_question, validate_medical_image

# Create blueprint
api = Blueprint('api', __name__, url_prefix='/api')


def _build_attention_visualization_base64(image, attention_map):
    """Convert a numeric attention map into a base64 PNG visualization."""
    attention_array = np.array(attention_map, dtype=np.float32)

    # Ensure map is normalized for stable rendering
    attention_array = (attention_array - attention_array.min()) / (
        attention_array.max() - attention_array.min() + 1e-8
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention overlay
    axes[1].imshow(image, alpha=0.7)
    axes[1].imshow(attention_array, cmap='jet', alpha=0.5)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64


@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'VisiHealth AI',
        'version': '1.0.0'
    })


@api.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    info = model_service.get_model_info()
    if info is None:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify(info)


@api.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expects:
        - image: Image file (multipart/form-data)
        - question: Question text (form field)
        
    Returns:
        - answer: Predicted answer
        - confidence: Confidence score
        - top_predictions: Top 3 predictions
        - roi: Region of interest info
        - rationale: Explanation
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'question' not in request.form:
            return jsonify({'error': 'No question provided'}), 400
        
        image_file = request.files['image']
        question = request.form['question']
        
        # Validate image
        is_valid, error = validate_image_file(image_file)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Validate question
        is_valid, error = validate_question(question)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Load image
        image = Image.open(image_file.stream).convert('RGB')
        
        # Validate that image looks medical (not a meme / photo)
        is_medical, medical_error = validate_medical_image(image)
        if not is_medical:
            return jsonify({'error': medical_error}), 422
        
        # Make prediction
        result = model_service.predict(image, question)

        # Normalize attention output to base64 image so frontend can render it directly.
        if 'attention_map' in result and isinstance(result['attention_map'], list):
            result['attention_map'] = _build_attention_visualization_base64(
                image,
                result['attention_map']
            )
        
        # Add metadata
        result['metadata'] = {
            'image_name': secure_filename(image_file.filename),
            'question': question,
            'image_size': image.size
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expects:
        - images: Multiple image files
        - questions: JSON array of questions
        
    Returns:
        - results: Array of predictions
    """
    try:
        # Check if images exist
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        if 'questions' not in request.form:
            return jsonify({'error': 'No questions provided'}), 400
        
        images = request.files.getlist('images')
        questions = request.form.getlist('questions')
        
        if len(images) != len(questions):
            return jsonify({'error': 'Number of images and questions must match'}), 400
        
        if len(images) > 10:
            return jsonify({'error': 'Maximum 10 images allowed per batch'}), 400
        
        # Process each image-question pair
        results = []
        for image_file, question in zip(images, questions):
            # Validate
            is_valid, error = validate_image_file(image_file)
            if not is_valid:
                results.append({'error': error})
                continue
            
            is_valid, error = validate_question(question)
            if not is_valid:
                results.append({'error': error})
                continue
            
            # Load and predict
            image = Image.open(image_file.stream).convert('RGB')
            result = model_service.predict(image, question)
            result['metadata'] = {
                'image_name': secure_filename(image_file.filename),
                'question': question
            }
            results.append(result)
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/visualize/attention', methods=['POST'])
def visualize_attention():
    """
    Generate attention map visualization
    
    Returns:
        - visualization: Base64 encoded image with attention overlay
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'question' not in request.form:
            return jsonify({'error': 'No question provided'}), 400
        
        image_file = request.files['image']
        question = request.form['question']
        
        # Validate
        is_valid, error = validate_image_file(image_file)
        if not is_valid:
            return jsonify({'error': error}), 400
        
        # Load image
        image = Image.open(image_file.stream).convert('RGB')
        
        # Validate that image looks medical
        is_medical, medical_error = validate_medical_image(image)
        if not is_medical:
            return jsonify({'error': medical_error}), 422
        
        # Get prediction with attention
        result = model_service.predict(image, question)
        
        if 'attention_map' not in result:
            return jsonify({'error': 'Attention map not available'}), 400

        # Reuse model output if already base64, otherwise convert from numeric map.
        if isinstance(result['attention_map'], str):
            img_base64 = result['attention_map']
        else:
            img_base64 = _build_attention_visualization_base64(image, result['attention_map'])
        
        return jsonify({
            'success': True,
            'data': {
                'visualization': img_base64,
                'answer': result['answer'],
                'confidence': result['confidence']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api.route('/answers/vocabulary', methods=['GET'])
def get_vocabulary():
    """Get all possible answers"""
    if model_service.answer_vocab is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'success': True,
        'data': {
            'vocabulary': list(model_service.answer_vocab.keys()),
            'total': len(model_service.answer_vocab)
        }
    })


# Error handlers
@api.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


@api.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500
