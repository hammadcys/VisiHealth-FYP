"""
VisiHealth AI - Flask Backend Application

This is the main Flask server that provides REST API endpoints
for the VisiHealth AI medical visual question answering system.

Architecture:
    - Flask web framework
    - RESTful API design
    - CORS enabled for frontend integration
    - Singleton pattern for model management
    - Error handling and validation

API Endpoints:
    - GET  /api/health - Health check
    - GET  /api/model/info - Model information
    - POST /api/predict - Single image prediction
    - POST /api/predict/batch - Batch predictions
    - POST /api/visualize/attention - Attention visualization
    - GET  /api/answers/vocabulary - Get answer vocabulary
"""

import os
import sys
from pathlib import Path
from flask import Flask, jsonify

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import config
from backend.api.routes import api
from backend.middleware.cors import init_cors
from backend.services.model_service import model_service


def create_app(config_name='default'):
    """
    Flask application factory
    
    Args:
        config_name: Configuration name (development/production)
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    app_config = config[config_name]()
    
    # Initialize CORS
    init_cors(app, app_config)
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Root endpoint
    @app.route('/')
    def index():
        """Root endpoint - API information"""
        return jsonify({
            'service': 'VisiHealth AI Backend',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'health': '/api/health',
                'predict': '/api/predict',
                'batch_predict': '/api/predict/batch',
                'attention': '/api/visualize/attention',
                'vocabulary': '/api/answers/vocabulary',
                'model_info': '/api/model/info'
            },
            'documentation': 'https://github.com/your-repo/visihealth-ai'
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    
    # Initialize model on application startup
    # Note: In Flask 3.0, @app.before_first_request is deprecated
    # So we load the model here during app creation
    try:
        print("\n" + "="*70)
        print(">>> VisiHealth AI Backend Server Starting...")
        print("="*70)
        
        model_service.load_model(
            config_path=app_config.MODEL_CONFIG_PATH,
            checkpoint_path=app_config.CHECKPOINT_PATH,
            model_info_path=app_config.MODEL_INFO_PATH,
            kg_file_path=app_config.KG_FILE_PATH
        )
        
        print("="*70)
        print(">>> Server Ready!")
        print(f"--- Listening on http://{app_config.HOST}:{app_config.PORT}")
        print(f"--- CORS enabled for: {app_config.CORS_ORIGINS}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"!!! Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return app


if __name__ == '__main__':
    # Get configuration from environment
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Create application
    app = create_app(env)
    
    # Get config
    app_config = config[env]()
    
    # Run server
    app.run(
        host=app_config.HOST,
        port=app_config.PORT,
        debug=app_config.DEBUG,
        threaded=True
    )
