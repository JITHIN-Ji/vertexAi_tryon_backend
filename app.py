import os
import json
import uuid
import base64
import tempfile
import logging
import typing
import threading
import time
from io import BytesIO
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image as PIL_Image
from google import genai
from google.genai.types import (
    Image,
    ProductImage,
    RecontextImageConfig,
    RecontextImageSource,
)
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Thread pool for async tasks
executor = ThreadPoolExecutor(max_workers=3)

# Store processing results (in production, use Redis or a database)
processing_results = {}
processing_lock = threading.Lock()

# Enable CORS
CORS(app, 
     origins=['*'], 
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/gif'}
VALID_CATEGORIES = ['upper_body', 'lower_body', 'dresses']

# Google AI Configuration
PROJECT_ID = os.getenv("PROJECT_ID", "poetic-chariot-471517-p8")
LOCATION = os.getenv("LOCATION", "us-central1")

# Initialize Google AI client
client = None
try:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
        service_account_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        client = genai.Client(
            vertexai=True, 
            project=PROJECT_ID, 
            location=LOCATION, 
            credentials=credentials
        )
        logger.info("Google AI client initialized successfully")
    else:
        logger.error("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")
except Exception as e:
    logger.error(f"❌ Failed to initialize Google AI client: {e}")
    client = None

def save_screenshot_to_supabase(image_path: str, request_id: str):
    """Save product screenshot to Supabase Storage via REST API - no supabase package needed"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            return

        from datetime import datetime
        import requests as req
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"products/{date_str}_{request_id}.jpg"

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        url = f"{supabase_url}/storage/v1/object/tryon-screenshots/{filename}"
        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "image/jpeg"
        }
        response = req.post(url, headers=headers, data=image_bytes, timeout=10)

        if response.status_code in (200, 201):
            logger.info(f"✅ Screenshot saved: {filename}")
        else:
            logger.warning(f"⚠️ Supabase returned {response.status_code}: {response.text}")

    except Exception as e:
        logger.warning(f"⚠️ Screenshot save failed (non-critical): {e}")

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file or not file.filename:
        return False, "No file provided"
    
    if not allowed_file(file.filename):
        return False, f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size by seeking to end
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return False, "File too large. Maximum size is 16MB"
    
    if file_size == 0:
        return False, "Empty file provided"
    
    return True, "Valid file"

def save_uploaded_file(file):
    """Save uploaded file to temporary location and return path"""
    try:
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        logger.info(f"File saved to: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return None

def crop_product_from_screenshot(image_path: str) -> str:
    """Crop the product photo area from Amazon screenshot for Vertex AI"""
    try:
        img = PIL_Image.open(image_path)
        width, height = img.size
        top = int(height * 0.20)
        bottom = int(height * 0.85)
        cropped = img.crop((0, top, width, bottom))
        cropped_path = image_path.replace('.jpg', '_cropped.jpg').replace('.jpeg', '_cropped.jpg')
        if cropped.mode != 'RGB':
            cropped = cropped.convert('RGB')
        cropped.save(cropped_path, 'JPEG', quality=90)
        logger.info(f"Cropped product area: {top}px to {bottom}px of {height}px total")
        return cropped_path
    except Exception as e:
        logger.warning(f"Crop failed, using original: {e}")
        return image_path

def pil_image_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    try:
        buffer = BytesIO()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format='PNG')
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise

def cleanup_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {e}")

def require_ai_client(f):
    """Decorator to check if AI client is available"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not client:
            return jsonify({
                'success': False,
                'error': 'Service unavailable - Google AI client not initialized',
                'message': 'The AI service is currently unavailable. Please try again later.'
            }), 503
        return f(*args, **kwargs)
    return decorated_function

def process_try_on_background(request_id, person_path, clothing_path, garment_description, category):
    """Process virtual try-on in background thread to prevent timeout"""
    cropped_path = None
    try:
        logger.info(f"[{request_id}] Starting background processing...")
        
        save_screenshot_to_supabase(clothing_path, request_id)
        
        # Crop product area for Vertex AI
        cropped_path = crop_product_from_screenshot(clothing_path)
        logger.info(f"[{request_id}] Using cropped image for try-on: {cropped_path}")
        
        # Call Google Gemini Virtual Try-On API
        response = client.models.recontext_image(
            model="virtual-try-on-001",
            source=RecontextImageSource(
                person_image=Image.from_file(location=person_path),
                product_images=[ProductImage(product_image=Image.from_file(location=cropped_path))],
            ),
            config=RecontextImageConfig(
                output_mime_type="image/png",
                number_of_images=1,
                safety_filter_level="BLOCK_LOW_AND_ABOVE",
            ),
        )
        
        logger.info(f"[{request_id}] Google AI API call successful!")
        
        # Process the generated image
        if not response.generated_images:
            with processing_lock:
                processing_results[request_id] = {
                    'status': 'failed',
                    'success': False,
                    'error': 'No image generated',
                    'message': 'The AI model did not generate any images. Please try again.'
                }
            logger.error(f"[{request_id}] No images generated by API")
            return
        
        result_image = typing.cast(PIL_Image.Image, response.generated_images[0].image._pil_image)
        output_base64 = pil_image_to_base64(result_image)
        
        with processing_lock:
            processing_results[request_id] = {
                'status': 'completed',
                'success': True,
                'message': 'Virtual try-on completed successfully',
                'results': {
                    'try_on_image': f"data:image/png;base64,{output_base64}",
                    'masked_image': None
                },
                'parameters': {
                    'garment_description': garment_description,
                    'category': category,
                    'model': 'virtual-try-on-001'
                },
                'completed_at': time.time()
            }
        
        logger.info(f"[{request_id}] Processing completed successfully")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error during background processing: {str(e)}")
        with processing_lock:
            processing_results[request_id] = {
                'status': 'failed',
                'success': False,
                'error': 'Virtual try-on failed',
                'message': f'An error occurred during processing: {str(e)}'
            }
    finally:
        # Always cleanup files including cropped version
        cleanup_files([person_path, clothing_path])
        if cropped_path and cropped_path != clothing_path:
            cleanup_files([cropped_path])

# Error handlers
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    logger.warning("File upload too large")
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'Maximum file size is 16MB'
    }), 413

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500

@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'message': 'The requested method is not allowed for this endpoint'
    }), 405

# Routes
@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    return jsonify({
        'message': 'Virtual Try-On API - Google Gemini Edition',
        'version': '1.0.0',
        'status': 'healthy' if client else 'unhealthy',
        'endpoints': {
            '/': {
                'method': 'GET',
                'description': 'API documentation and status'
            },
            '/health': {
                'method': 'GET',
                'description': 'Check API health status'
            },
            '/try-on': {
                'method': 'POST',
                'description': 'Upload person and clothing images for virtual try-on',
                'parameters': {
                    'person_image': 'Image file of person (required)',
                    'clothing_image': 'Image file of clothing (required)',
                    'garment_description': 'Description of the garment (optional, default: "stylish clothing")',
                    'category': 'Category: upper_body, lower_body, or dresses (optional, default: "upper_body")'
                },
                'accepted_formats': list(ALLOWED_EXTENSIONS),
                'max_file_size': '16MB'
            }
        },
        'model': 'virtual-try-on-001'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = 'healthy' if client else 'unhealthy'
    message = 'API is running' if client else 'Google AI client not initialized'
    
    return jsonify({
        'status': status,
        'message': message,
        'model': 'virtual-try-on-001',
        'timestamp': str(int(__import__('time').time())),
        'version': '1.0.0'
    }), 200 if client else 503

@app.route('/try-on', methods=['POST'])
@require_ai_client
def virtual_try_on():
    """Main virtual try-on endpoint - starts async processing"""
    person_path = None
    clothing_path = None
    request_id = str(uuid.uuid4())
    
    try:
        # Validate request
        if 'person_image' not in request.files or 'clothing_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Missing required files',
                'message': 'Both person_image and clothing_image are required'
            }), 400
        
        person_image = request.files['person_image']
        clothing_image = request.files['clothing_image']
        garment_description = request.form.get('garment_description', 'stylish clothing')
        category = request.form.get('category', 'upper_body')
        
        # Validate category
        if category not in VALID_CATEGORIES:
            return jsonify({
                'success': False,
                'error': 'Invalid category',
                'message': f'Category must be one of: {", ".join(VALID_CATEGORIES)}'
            }), 400
        
        # Validate person image
        is_valid, message = validate_image_file(person_image)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid person image',
                'message': message
            }), 400
        
        # Validate clothing image
        is_valid, message = validate_image_file(clothing_image)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid clothing image',
                'message': message
            }), 400
        
        # Save files
        person_path = save_uploaded_file(person_image)
        clothing_path = save_uploaded_file(clothing_image)
        
        if not person_path or not clothing_path:
            return jsonify({
                'success': False,
                'error': 'File processing error',
                'message': 'Failed to process uploaded files'
            }), 500
        
        logger.info(f"[{request_id}] Processing try-on with category: {category}, description: {garment_description}")
        
        # Start async background processing
        with processing_lock:
            processing_results[request_id] = {
                'status': 'processing',
                'started_at': time.time()
            }
        
        executor.submit(
            process_try_on_background,
            request_id,
            person_path,
            clothing_path,
            garment_description,
            category
        )
        
        # Return immediately with request ID for polling
        return jsonify({
            'success': True,
            'message': 'Processing started',
            'request_id': request_id,
            'status_url': f'/try-on/status/{request_id}',
            'note': 'Poll the status_url endpoint to get the result'
        }), 202
        
    except Exception as e:
        # Clean up files in case of error
        cleanup_files([person_path, clothing_path])
        
        logger.error(f"[{request_id}] Error during try-on request: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Request processing failed',
            'message': f'An error occurred: {str(e)}'
        }), 500

@app.route('/try-on/status/<request_id>', methods=['GET'])
def try_on_status(request_id):
    """Check the status of a virtual try-on request"""
    with processing_lock:
        if request_id not in processing_results:
            return jsonify({
                'success': False,
                'error': 'Request not found',
                'message': 'The specified request ID was not found or has expired'
            }), 404
        
        result = processing_results[request_id].copy()
    
    # If still processing, return 202 Accepted
    if result.get('status') == 'processing':
        elapsed = time.time() - result.get('started_at', time.time())
        return jsonify({
            'success': True,
            'status': 'processing',
            'message': f'Still processing ({elapsed:.1f}s elapsed)',
            'request_id': request_id
        }), 202
    
    # If completed, remove from cache and return full result
    if result.get('status') == 'completed':
        with processing_lock:
            processing_results.pop(request_id, None)
        return jsonify(result), 200
    
    # If failed, return error
    if result.get('status') == 'failed':
        with processing_lock:
            processing_results.pop(request_id, None)
        return jsonify(result), 500
    
    return jsonify({
        'success': False,
        'error': 'Unknown status',
        'message': 'The request status is unknown'
    }), 500

@app.route('/favicon.ico')
def favicon():
    """Favicon endpoint to prevent 404 errors"""
    return '', 204

# Production WSGI application
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    if not debug:
        logger.info("Starting production server...")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )