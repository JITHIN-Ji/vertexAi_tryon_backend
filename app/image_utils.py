import os
import uuid
from io import BytesIO

from werkzeug.utils import secure_filename
from PIL import Image as PIL_Image
import cv2

from app.config import app, logger, ALLOWED_EXTENSIONS
from app.garment_extractor import extract_garment_by_class, composite_on_bg
import base64


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file):
    if not file or not file.filename:
        return False, "No file provided"
    if not allowed_file(file.filename):
        return False, f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return False, "File too large. Maximum size is 16MB"
    if file_size == 0:
        return False, "Empty file provided"
    return True, "Valid file"


def save_uploaded_file(file):
    try:
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        logger.info(f"File saved to: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return None


def prepare_garment_image(request_id: str, clothing_path: str, garment_class: str) -> str:
    """
    Runs DeepFashion2 (YOLO) on the screenshot, isolates the garment Gemini
    identified (garment_class — falls back to largest-area mask if that
    class wasn't detected), composites it on a clean off-white background,
    and returns the path to THAT image for try-on.

    If the model isn't loaded or nothing is detected, falls back to the
    original full screenshot (no cropping logic at all anymore).
    """
    from app.ai_clients import YOLO_MODEL

    if YOLO_MODEL is None:
        logger.warning(f"[{request_id}] YOLO model not loaded — using original screenshot")
        return clothing_path

    try:
        extraction = extract_garment_by_class(
            clothing_path, YOLO_MODEL, garment_class=garment_class or None
        )
        garment_rgba = extraction.get("garment_image")

        if garment_rgba is None:
            logger.warning(f"[{request_id}] No garment detected by YOLO — using original screenshot")
            return clothing_path

        logger.info(
            f"[{request_id}] DeepFashion2 isolated class={extraction['class_name']} "
            f"used_fallback={extraction['used_fallback']}"
        )

        bg_image = composite_on_bg(garment_rgba)
        extracted_path = (
            clothing_path.replace('.jpg', '_garment.jpg').replace('.jpeg', '_garment.jpg')
        )
        cv2.imwrite(extracted_path, bg_image)
        return extracted_path

    except Exception as e:
        logger.warning(f"[{request_id}] DeepFashion2 extraction failed ({e}) — using original screenshot")
        return clothing_path


def pil_image_to_base64(pil_image):
    try:
        buffer = BytesIO()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise


def cleanup_files(file_paths):
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {e}")