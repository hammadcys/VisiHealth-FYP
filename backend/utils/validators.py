"""
VisiHealth AI - Input Validators
"""

from pathlib import Path
import numpy as np
from PIL import Image as PILImage


def validate_image_file(file):
    """
    Validate uploaded image file
    
    Args:
        file: Flask FileStorage object
        
    Returns:
        (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'dcm'}
    file_ext = Path(file.filename).suffix.lower().lstrip('.')
    
    if file_ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
    
    return True, None


def validate_medical_image(image: PILImage.Image):
    """
    Heuristically decide whether an uploaded image looks medical.

    Medical images (X-rays, MRIs, CT scans, ultrasounds, histology slides)
    share a distinctive signature: they are predominantly grayscale *or* have
    very limited, desaturated colour.  Memes, selfies, and general photographs
    contain large areas of vivid, saturated colour.

    Strategy
    --------
    1. Convert to HSV and compute the *saturation* channel (0-255).
    2. A pixel is considered "highly coloured" when its saturation > 60 / 255.
    3. If more than 35 % of pixels are highly coloured → likely NOT medical.
    4. Additional guard: images that are almost entirely one flat colour
       (e.g. solid-colour test images) are also rejected.

    Returns
    -------
    (is_medical: bool, error_message: str | None)
    """
    try:
        # Work on a small thumbnail for speed (max 256 px on longest side)
        thumb = image.copy()
        thumb.thumbnail((256, 256), PILImage.LANCZOS)
        rgb = thumb.convert("RGB")
        arr = np.array(rgb, dtype=np.float32)  # shape (H, W, 3)

        # --- Saturation via HSV --------------------------------------------------
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Saturation in [0, 1]  (0 where cmax == 0)
        sat = np.where(cmax > 0, delta / (cmax + 1e-6), 0.0)

        highly_coloured_ratio = float(np.mean(sat > (60.0 / 255.0)))

        # --- Flat / blank image guard -------------------------------------------
        # Reject images where the entire image is one uniform colour (test images)
        pixel_std = float(np.std(arr))

        if pixel_std < 5.0:
            return False, (
                "The uploaded image appears to be blank or a solid colour. "
                "Please upload a real medical image (X-ray, MRI, CT scan, etc.)."
            )

        # --- Primary check -------------------------------------------------------
        if highly_coloured_ratio > 0.35:
            return False, (
                "This does not appear to be a medical image. "
                "Please upload a medical scan such as an X-ray, MRI, CT scan, "
                "ultrasound, or histology slide."
            )

        return True, None

    except Exception:
        # If analysis fails for any reason, let the image through
        # (don't block users due to a validator bug).
        return True, None


def validate_question(question):
    """
    Validate question text
    
    Args:
        question: Question string
        
    Returns:
        (is_valid, error_message)
    """
    if not question or not isinstance(question, str):
        return False, "Question must be a non-empty string"
    
    question = question.strip()
    
    if len(question) < 3:
        return False, "Question is too short (minimum 3 characters)"
    
    if len(question) > 500:
        return False, "Question is too long (maximum 500 characters)"
    
    return True, None


def sanitize_filename(filename):
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove path components
    filename = Path(filename).name
    
    # Remove potentially dangerous characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    filename = ''.join(c if c in safe_chars else '_' for c in filename)
    
    return filename
