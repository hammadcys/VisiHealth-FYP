"""
VisiHealth AI - Backend Configuration
"""

import os
from pathlib import Path

class Config:
    """Flask application configuration"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    BACKEND_DIR = Path(__file__).parent
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'visihealth-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    
    # Server settings
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Upload settings
    UPLOAD_FOLDER = BACKEND_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'dcm'}
    
    # Model settings
    MODEL_CONFIG_PATH = BASE_DIR / 'config.yaml'
    CHECKPOINT_PATH = BASE_DIR / 'checkpoints' / 'best_checkpoint.pth'
    MODEL_INFO_PATH = BASE_DIR / 'results' / 'VisiHealth_Model_Info.json'
    KG_FILE_PATH = BASE_DIR / 'kg.txt'
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # Response settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # Ensure upload folder exists
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # In production, SECRET_KEY should be set via environment variable
    # Otherwise inherits from Config class
    if os.environ.get('SECRET_KEY'):
        SECRET_KEY = os.environ.get('SECRET_KEY')
    # If SECRET_KEY not in environment, it will use base Config's default


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
