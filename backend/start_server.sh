#!/bin/bash
# VisiHealth AI - Flask Backend Launcher (Linux/Mac)

echo "============================================================"
echo "VisiHealth AI - Starting Backend Server"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo "Installing/Updating dependencies..."
pip3 install -r requirements.txt --quiet

echo ""
echo "Starting Flask server..."
echo "Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

# Set environment variables
export FLASK_ENV=development
export FLASK_DEBUG=True

# Start Flask app
python3 app.py
