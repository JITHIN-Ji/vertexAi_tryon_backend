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
import google.generativeai as gemini_ai

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

# Google Vertex AI Configuration
PROJECT_ID = os.getenv("PROJECT_ID", "poetic-chariot-471517-p8")
LOCATION = os.getenv("LOCATION", "us-central1")

# Gemini API key (for validation + extract-info)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini validation prompt
GEMINI_PROMPT = """
You are a garment validator for a virtual try-on app running on Amazon product pages.

You will receive a screenshot from the Amazon mobile app. Follow these steps strictly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — IS THIS AN AMAZON PRODUCT PAGE WITH A CLOTHING ITEM?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is always an Amazon product page screenshot. Check if the main product being sold is a clothing/wearable garment.

Accepted clothing types (answer YES to Step 1):
- Shirts, t-shirts, tops, blouses, kurtas, sarees, lehengas, kurtis
- Pants, jeans, trousers, shorts, skirts, palazzos
- Dresses, suits, jackets, coats, hoodies, sweaters, cardigans
- Ethnic wear, sportswear, activewear, innerwear, swimwear, nightwear

NOT clothing — reject these (answer NO to Step 1):
- Shoes, sandals, slippers, boots (footwear only)
- Bags, purses, wallets, backpacks (no clothing shown)
- Jewelry, watches, sunglasses (accessories only)
- Electronics, home decor, furniture, kitchen items, books, food
- A search results page / category grid showing many products

If Step 1 = NO → reply exactly: NO_GARMENT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — IDENTIFY THE PRIMARY GARMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
On this product page, there may be ONE main product photo shown large, and other items visible partially (e.g. a model wearing a shirt while showing pants, or a small accessory in the corner).

Your job: identify which garment occupies the MOST space / is most dominant in the image.

Rules:
- The garment that takes up the largest area of the image is the PRIMARY garment.
- Ignore partially visible items (less than 30% of image area).
- If a model is wearing pants and only their torso/shirt is barely visible at the top edge, the PRIMARY garment is the pants.
- If a model is wearing a t-shirt and the pants are only partially visible at the bottom, the PRIMARY garment is the t-shirt.
- Focus only on the ONE dominant garment for all further checks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — IS THE PRIMARY GARMENT CLEARLY VISIBLE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Check the PRIMARY garment identified in Step 2.

PASSES (reply READY):
- At least 60% of the garment is visible in the screenshot
- It is shown front-facing or at a slight angle
- It is on a model, mannequin, hanger, or flat-lay
- It is reasonably well-lit and not severely blurry
- NOTE: In Amazon mobile screenshots, UI elements naturally take space and models are often partially visible — this is normal and acceptable as long as 60% of the garment is visible

PARTIAL (reply PARTIAL_GARMENT):
- The product IS a clothing item
- The garment type CAN be identified (you can tell what type of clothing it is)
- BUT less than 60% of the garment is visible in the screenshot
- Examples of PARTIAL_GARMENT:
  * Shirt visible only from chest to neck — bottom half missing
  * Pants visible only below the knees — upper part missing
  * Dress where only the top 30% is showing
  * Only a sleeve or collar visible with no body of the garment
  * Zoomed in so close that less than 60% of the full garment is in frame

FAILS (reply UNCLEAR_GARMENT):
- The screenshot shows a REVIEWS section (star ratings, customer review text, "Top reviews" heading)
- The screenshot shows a DESCRIPTION / ABOUT section (bullet points of product features, text only)
- The screenshot shows a SIZE CHART section (measurement tables, size grids)
- The screenshot shows a QUESTIONS & ANSWERS section (Q&A text content)
- The screenshot shows a SPONSORED / RECOMMENDATIONS section (multiple small product thumbnails)
- The screenshot shows a DELIVERY / RETURNS section (shipping info, return policy text)
- The primary garment is tiny or less than 30% of the image
- The image is severely blurry, pixelated, or too dark to see garment details
- The garment is only visible from the back with no front detail at all

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE RULES — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reply with ONLY one of these four — no other text, no explanation, no punctuation:
    NO_GARMENT
    UNCLEAR_GARMENT
    PARTIAL_GARMENT
    READY
""".strip()

# Initialize Vertex AI client
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


# ─────────────────────────────────────────────────────────────
#  SUPABASE HELPERS
# ─────────────────────────────────────────────────────────────

def save_screenshot_to_supabase(image_path: str, request_id: str):
    """Save READY screenshots to products/ folder in Supabase"""
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


def save_failed_screenshot_to_supabase(image_path: str, reason: str, request_id: str):
    """Save FAILED/rejected screenshots to rejected/{reason}/ folder in Supabase"""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            return

        from datetime import datetime
        import requests as req
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"rejected/{reason}/{date_str}_{request_id}.jpg"

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        url = f"{supabase_url}/storage/v1/object/tryon-screenshots/{filename}"
        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "image/jpeg"
        }
        response = req.post(url, headers=headers, data=image_bytes, timeout=10)

        if response.status_code in (200, 201):
            logger.info(f"✅ Failed screenshot saved: {filename}")
        else:
            logger.warning(f"⚠️ Supabase save failed: {response.status_code}: {response.text}")

    except Exception as e:
        logger.warning(f"⚠️ Failed screenshot save error (non-critical): {e}")


# ─────────────────────────────────────────────────────────────
#  FILE HELPERS
# ─────────────────────────────────────────────────────────────

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


def require_ai_client(f):
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


# ─────────────────────────────────────────────────────────────
#  BACKGROUND PROCESSING
# ─────────────────────────────────────────────────────────────

def process_try_on_background(request_id, person_path, clothing_path, garment_description, category):
    """Process virtual try-on in background thread"""
    cropped_path = None
    try:
        logger.info(f"[{request_id}] Starting background processing...")

        # Crop product area for Vertex AI
        cropped_path = crop_product_from_screenshot(clothing_path)
        logger.info(f"[{request_id}] Using cropped image for try-on: {cropped_path}")

        # Call Vertex AI Virtual Try-On
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
        cleanup_files([person_path, clothing_path])
        if cropped_path and cropped_path != clothing_path:
            cleanup_files([cropped_path])


# ─────────────────────────────────────────────────────────────
#  ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'success': False, 'error': 'File too large', 'message': 'Maximum file size is 16MB'}), 413

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found', 'message': 'The requested endpoint does not exist'}), 404

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error', 'message': 'An unexpected error occurred.'}), 500

@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({'success': False, 'error': 'Method not allowed', 'message': 'The requested method is not allowed for this endpoint'}), 405


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Virtual Try-On API - Google Gemini Edition',
        'version': '1.0.0',
        'status': 'healthy' if client else 'unhealthy',
        'endpoints': {
            '/validate':     {'method': 'POST', 'description': 'Validate screenshot for clothing'},
            '/extract-info': {'method': 'POST', 'description': 'Extract product name and brand'},
            '/try-on':       {'method': 'POST', 'description': 'Start virtual try-on'},
            '/try-on/status/<id>': {'method': 'GET', 'description': 'Poll try-on result'},
            '/health':       {'method': 'GET',  'description': 'Health check'},
        },
        'model': 'virtual-try-on-001'
    })


@app.route('/health', methods=['GET'])
def health_check():
    status = 'healthy' if client else 'unhealthy'
    return jsonify({
        'status': status,
        'message': 'API is running' if client else 'Google AI client not initialized',
        'model': 'virtual-try-on-001',
        'timestamp': str(int(time.time())),
        'version': '1.0.0'
    }), 200 if client else 503


# ─────────────────────────────────────────────────────────────
#  NEW: /validate
# ─────────────────────────────────────────────────────────────

@app.route('/validate', methods=['POST'])
def validate_garment():
    """Validate screenshot with Gemini — saves screenshots to Supabase"""
    image_path = None
    request_id = str(uuid.uuid4())
    try:
        if 'screenshot' not in request.files:
            return jsonify({'result': 'ERROR', 'message': 'No screenshot provided'}), 400

        screenshot = request.files['screenshot']
        is_valid, msg = validate_image_file(screenshot)
        if not is_valid:
            return jsonify({'result': 'ERROR', 'message': msg}), 400

        image_path = save_uploaded_file(screenshot)
        if not image_path:
            return jsonify({'result': 'ERROR', 'message': 'Failed to save file'}), 500

        # Call Gemini
        gemini_ai.configure(api_key=GEMINI_API_KEY)
        model = gemini_ai.GenerativeModel("gemini-2.5-flash")

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()},
            GEMINI_PROMPT
        ])

        result_text = response.text.strip().upper().replace(".", "").replace("\n", "")
        logger.info(f"[{request_id}] Gemini result: {result_text}")

        if "NO_GARMENT" in result_text:
            result = "NO_GARMENT"
        elif "UNCLEAR_GARMENT" in result_text:
            result = "UNCLEAR_GARMENT"
        elif "PARTIAL_GARMENT" in result_text:
            result = "PARTIAL_GARMENT"
        elif "READY" in result_text:
            result = "READY"
        else:
            result = "UNCLEAR_GARMENT"

        # Save only failed/rejected screenshots to Supabase
        if result != "READY":
            save_failed_screenshot_to_supabase(image_path, result, request_id)

        return jsonify({
            'result': result,
            'request_id': request_id
        }), 200

    except Exception as e:
        logger.error(f"Validate error: {e}")
        if image_path:
            save_failed_screenshot_to_supabase(image_path, "ERROR", request_id)
        return jsonify({'result': 'ERROR', 'message': str(e)}), 500

    finally:
        cleanup_files([image_path])


# ─────────────────────────────────────────────────────────────
#  NEW: /extract-info
# ─────────────────────────────────────────────────────────────

@app.route('/extract-info', methods=['POST'])
def extract_product_info():
    """Extract product name and brand from Amazon screenshot"""
    image_path = None
    try:
        if 'screenshot' not in request.files:
            return jsonify({'product_name': 'Unknown', 'brand': 'Unknown'}), 200

        screenshot = request.files['screenshot']
        image_path = save_uploaded_file(screenshot)
        if not image_path:
            return jsonify({'product_name': 'Unknown', 'brand': 'Unknown'}), 200

        gemini_ai.configure(api_key=GEMINI_API_KEY)
        model = gemini_ai.GenerativeModel("gemini-3.1-pro-preview")

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        prompt = """
You are a product information extractor for a shopping app screenshot.
Extract:
1. PRODUCT_NAME: The clothing product name/title (max 6 words)
2. BRAND: The brand or seller name

If not found write UNKNOWN.
Reply in EXACTLY this format only:
PRODUCT_NAME: <name>
BRAND: <brand>
""".strip()

        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()},
            prompt
        ])

        text = response.text.strip()
        product_name = "Unknown"
        brand = "Unknown"

        for line in text.splitlines():
            if line.startswith("PRODUCT_NAME:"):
                product_name = line.split("PRODUCT_NAME:", 1)[1].strip() or "Unknown"
            elif line.startswith("BRAND:"):
                brand = line.split("BRAND:", 1)[1].strip() or "Unknown"

        logger.info(f"Extracted: {product_name} / {brand}")
        return jsonify({'product_name': product_name, 'brand': brand}), 200

    except Exception as e:
        logger.error(f"Extract info error: {e}")
        return jsonify({'product_name': 'Unknown', 'brand': 'Unknown'}), 200

    finally:
        cleanup_files([image_path])


# ─────────────────────────────────────────────────────────────
#  EXISTING: /try-on
# ─────────────────────────────────────────────────────────────

@app.route('/try-on', methods=['POST'])
@require_ai_client
def virtual_try_on():
    """Main virtual try-on endpoint - starts async processing"""
    person_path = None
    clothing_path = None
    request_id = str(uuid.uuid4())

    try:
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

        if category not in VALID_CATEGORIES:
            return jsonify({
                'success': False,
                'error': 'Invalid category',
                'message': f'Category must be one of: {", ".join(VALID_CATEGORIES)}'
            }), 400

        is_valid, message = validate_image_file(person_image)
        if not is_valid:
            return jsonify({'success': False, 'error': 'Invalid person image', 'message': message}), 400

        is_valid, message = validate_image_file(clothing_image)
        if not is_valid:
            return jsonify({'success': False, 'error': 'Invalid clothing image', 'message': message}), 400

        person_path = save_uploaded_file(person_image)
        clothing_path = save_uploaded_file(clothing_image)

        if not person_path or not clothing_path:
            return jsonify({'success': False, 'error': 'File processing error', 'message': 'Failed to process uploaded files'}), 500

        logger.info(f"[{request_id}] Processing try-on: category={category}")

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

        return jsonify({
            'success': True,
            'message': 'Processing started',
            'request_id': request_id,
            'status_url': f'/try-on/status/{request_id}',
            'note': 'Poll the status_url endpoint to get the result'
        }), 202

    except Exception as e:
        cleanup_files([person_path, clothing_path])
        logger.error(f"[{request_id}] Error: {str(e)}")
        return jsonify({'success': False, 'error': 'Request processing failed', 'message': str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  EXISTING: /try-on/status
# ─────────────────────────────────────────────────────────────

@app.route('/try-on/status/<request_id>', methods=['GET'])
def try_on_status(request_id):
    with processing_lock:
        if request_id not in processing_results:
            return jsonify({
                'success': False,
                'error': 'Request not found',
                'message': 'The specified request ID was not found or has expired'
            }), 404
        result = processing_results[request_id].copy()

    if result.get('status') == 'processing':
        elapsed = time.time() - result.get('started_at', time.time())
        return jsonify({
            'success': True,
            'status': 'processing',
            'message': f'Still processing ({elapsed:.1f}s elapsed)',
            'request_id': request_id
        }), 202

    if result.get('status') == 'completed':
        with processing_lock:
            processing_results.pop(request_id, None)
        return jsonify(result), 200

    if result.get('status') == 'failed':
        with processing_lock:
            processing_results.pop(request_id, None)
        return jsonify(result), 500

    return jsonify({'success': False, 'error': 'Unknown status'}), 500


@app.route('/favicon.ico')
def favicon():
    return '', 204


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    if not debug:
        logger.info("Starting production server...")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)