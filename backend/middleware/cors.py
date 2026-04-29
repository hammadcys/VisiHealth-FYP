"""
VisiHealth AI - CORS Middleware
"""

from flask_cors import CORS


def init_cors(app, config):
    """
    Initialize CORS for the Flask app
    
    Args:
        app: Flask application
        config: Configuration object
    """
    CORS(app, 
         origins=config.CORS_ORIGINS,
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization'],
         supports_credentials=True,
         max_age=3600)
    
    return app
