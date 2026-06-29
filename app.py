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
from google import genai as gemini_ai
from google.genai import types as genai_types
from analytics import create_session, update_validate, update_tryon, save_feedback
import cv2
from garment_extractor import extract_garment_by_class, composite_on_bg
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

executor = ThreadPoolExecutor(max_workers=3)
processing_results = {}
processing_lock = threading.Lock()

def cleanup_old_results():
    while True:
        time.sleep(60)
        now = time.time()
        with processing_lock:
            expired = [
                rid for rid, r in processing_results.items()
                if r.get('status') in ('completed', 'failed')
                and now - r.get('completed_at', now) > 60
            ]
            for rid in expired:
                processing_results.pop(rid, None)
                logger.info(f"Cleaned up expired result: {rid}")

threading.Thread(target=cleanup_old_results, daemon=True).start()

CORS(app,
     origins=['*'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/gif'}
VALID_CATEGORIES = ['upper_body', 'lower_body', 'dresses']

PROJECT_ID = os.getenv("PROJECT_ID", "poetic-chariot-471517-p8")
LOCATION = os.getenv("LOCATION", "us-central1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VALID_GARMENT_CLASSES = {
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", "skirt",
    "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
}

GEMINI_PROMPT = """
You are a garment validator AND product-info extractor for a virtual try-on app
running on fashion shopping apps (Amazon, Flipkart, Myntra, Meesho).

You are given TWO inputs:
1. A SCREENSHOT image of a product page.
2. A block of ACCESSIBILITY TEXT NODES extracted from the same screen, in the
   order they appear top-to-bottom (this includes the product title, brand,
   price, and other on-screen text — use it to read text that may be small,
   cut off, or stylised in the image).

Do the following steps IN ORDER.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — IS THIS A FASHION APP PRODUCT PAGE WITH A CLOTHING ITEM?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Check if the main product being sold is a clothing/wearable garment.
Accepted (YES): shirts, t-shirts, tops, blouses, kurtas, sarees, lehengas,
kurtis, pants, jeans, trousers, shorts, skirts, palazzos, dresses, suits,
jackets, coats, hoodies, sweaters, cardigans, ethnic wear, sportswear,
activewear, innerwear, swimwear, nightwear.
NOT clothing (NO): shoes/sandals/boots, bags/wallets, jewelry/watches/
sunglasses, electronics, home/kitchen/furniture, books, food, or a search
results/category grid showing many products.

If Step 1 = NO → result = "NO_GARMENT", set product_title, brand_name,
garment_class to empty strings, skip Steps 2-5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — IDENTIFY THE PRIMARY GARMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
There may be ONE main product photo shown large, plus other items partially
visible. The PRIMARY garment is whichever takes up the most area / is most
dominant in the image. Ignore items under 30% of image area. Focus only on
this ONE garment for everything below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — IS THE PRIMARY GARMENT CLEARLY VISIBLE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
READY: at least 60% of the garment visible, front-facing or slight angle,
on model/mannequin/hanger/flat-lay, reasonably lit, not severely blurry.
PARTIAL_GARMENT: it IS clothing and you CAN tell the type, but less than 60%
of it is visible (e.g. only chest-to-neck of a shirt, only below-knee of
pants, only top 30% of a dress).
UNCLEAR_GARMENT: page shows reviews / description-bullets / size chart /
Q&A / sponsored-recommendations grid / delivery-returns text instead of the
product photo, OR garment is under 30% of image, OR image too blurry/dark,
OR only the back is visible with zero front detail.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — EXTRACT PRODUCT TITLE AND BRAND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Using BOTH the accessibility text nodes and the image:
- product_title: the full product title exactly as shown on the page
  (e.g. "Amazon Brand - Symbol Men's Cotton Shirt | Chinese Collar |
  Casual"). It's usually the longest descriptive line near the top of the
  text nodes or right under the brand link.
- brand_name: the brand/seller name only (e.g. "Symbol", "Roadster", "H&M"),
  often the node like "Visit the Symbol Store".
Never invent text. If you can't confidently find one, return "" for it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — CLASSIFY THE PRIMARY GARMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pick EXACTLY ONE class from this fixed list using both the image and the
title/text (pick whichever is the PRIMARY garment from Step 2, even on a
combo/co-ord page):
  short_sleeved_shirt, long_sleeved_shirt, short_sleeved_outwear,
  long_sleeved_outwear, vest, sling, shorts, trousers, skirt,
  short_sleeved_dress, long_sleeved_dress, vest_dress, sling_dress
Guidance: kurti/kurta/tunic/blouse → shirt classes by sleeve length.
Sarees/anarkalis/gowns/lehengas → whichever dress/skirt class best matches
silhouette + sleeve length. If result is NO_GARMENT, garment_class = "".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reply with ONLY a single-line JSON object — no markdown, no code fences, no
explanation:
{"result": "...", "product_title": "...", "brand_name": "...", "garment_class": "..."}
""".strip()

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


YOLO_MODEL = None
try:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    _model_path = hf_hub_download(
        repo_id="Bingsu/adetailer",
        filename="deepfashion2_yolov8s-seg.pt",
    )
    YOLO_MODEL = YOLO(_model_path)
    logger.info("✅ DeepFashion2 YOLO model loaded")
except Exception as e:
    logger.error(f"❌ Failed to load DeepFashion2 model: {e}")
    YOLO_MODEL = None


SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "amazon_screenshot_garmentimages")

def save_tryon_images_to_supabase(
    garment_bytes: bytes,
    result_base64: str,
    session_id: str,
    user_id: str
):
    try:
        import requests as req
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            logger.warning("⚠️ Supabase credentials not set, skipping save")
            return

        headers = {"Authorization": f"Bearer {supabase_key}"}
        base_path = f"tryon-results/{user_id}/{session_id}"

        # 1. Upload garment screenshot
        garment_url = f"{supabase_url}/storage/v1/object/{SUPABASE_BUCKET}/{base_path}/garment.jpg?upsert=true"
        r1 = req.post(garment_url, headers={**headers, "Content-Type": "image/jpeg"},
                      data=garment_bytes, timeout=30)  # ← was 10
        if r1.status_code in (200, 201):
            logger.info(f"✅ Garment saved: {base_path}/garment.jpg")
        else:
            logger.warning(f"⚠️ Garment upload: {r1.status_code} - {r1.text}")

        # 2. Upload try-on result
        result_bytes = base64.b64decode(result_base64)
        logger.info(f"Uploading result PNG: {len(result_bytes) / 1024:.1f} KB")  # ← add size log
        result_url = f"{supabase_url}/storage/v1/object/{SUPABASE_BUCKET}/{base_path}/result.png?upsert=true"
        r2 = req.post(result_url, headers={**headers, "Content-Type": "image/png"},
                      data=result_bytes, timeout=60)  # ← was 10, PNG is much larger
        if r2.status_code in (200, 201):
            logger.info(f"✅ Result saved: {base_path}/result.png")
        else:
            logger.warning(f"⚠️ Result upload: {r2.status_code} - {r2.text}")

    except Exception as e:
        logger.warning(f"⚠️ save_tryon_images_to_supabase failed (non-critical): {e}")


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


def parse_gemini_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
    return {}


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




def process_try_on_background(request_id, person_path, clothing_path,
                               garment_description, category,
                               session_id, user_id, garment_class=""):
    cropped_path = None
    start_ms = int(time.time() * 1000)
    try:
        logger.info(f"[{request_id}] garment_class hint from validate: {garment_class!r}")
        cropped_path = prepare_garment_image(request_id, clothing_path, garment_class)


        response = client.models.recontext_image(
            model="virtual-try-on-001",
            source=RecontextImageSource(
                person_image=Image.from_file(location=person_path),
                product_images=[ProductImage(product_image=Image.from_file(location=cropped_path))],
            ),
            config=RecontextImageConfig(
                output_mime_type="image/png",
                number_of_images=1,
                safety_filter_level="BLOCK_ONLY_HIGH",
                person_generation="allow_all",
            ),
        )

        logger.info(f"[{request_id}] Google AI API call successful!")

        if not response.generated_images:
            elapsed_ms = int(time.time() * 1000) - start_ms
            update_tryon(session_id, "failed", elapsed_ms, "No image generated")
            with processing_lock:
                processing_results[request_id] = {
                    'status': 'failed',
                    'success': False,
                    'error': 'No image generated',
                    'message': 'The AI model did not generate any images. Please try again.',
                    'completed_at': time.time()
                }
            return

        result_image = typing.cast(PIL_Image.Image, response.generated_images[0].image._pil_image)
        output_base64 = pil_image_to_base64(result_image)

        elapsed_ms = int(time.time() * 1000) - start_ms

        garment_image_path = f"tryon-results/{user_id}/{session_id}/garment.jpg"
        result_image_path  = f"tryon-results/{user_id}/{session_id}/result.png"

        # Read garment bytes BEFORE cleanup so background upload still has data
        with open(clothing_path, 'rb') as f:
            garment_bytes = f.read()

        
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

        
        def save_in_background():
            save_tryon_images_to_supabase(
                garment_bytes=garment_bytes,
                result_base64=output_base64,
                session_id=session_id,
                user_id=user_id
            )
            update_tryon(session_id, "success", elapsed_ms,
                        garment_image_path=garment_image_path,
                        result_image_path=result_image_path)

        executor.submit(save_in_background)

        logger.info(f"[{request_id}] Processing completed successfully")

    except Exception as e:
        elapsed_ms = int(time.time() * 1000) - start_ms
        update_tryon(session_id, "failed", elapsed_ms, str(e))
        logger.error(f"[{request_id}] Error during background processing: {str(e)}")
        with processing_lock:
            processing_results[request_id] = {
                'status': 'failed',
                'success': False,
                'error': 'Virtual try-on failed',
                'message': f'An error occurred during processing: {str(e)}',
                'completed_at': time.time()
            }
    finally:
        cleanup_files([person_path, clothing_path])
        if cropped_path and cropped_path != clothing_path:
            cleanup_files([cropped_path])




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



@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Virtual Try-On API - Google Gemini Edition',
        'version': '1.0.0',
        'status': 'healthy' if client else 'unhealthy',
        'endpoints': {
            '/validate':           {'method': 'POST', 'description': 'Validate screenshot for clothing'},
            '/try-on':             {'method': 'POST', 'description': 'Start virtual try-on'},
            '/try-on/status/<id>': {'method': 'GET',  'description': 'Poll try-on result'},
            '/health':             {'method': 'GET',  'description': 'Health check'},
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




@app.route('/validate', methods=['POST'])
def validate_garment():
    image_path = None
    request_id = str(uuid.uuid4())
    user_id = request.headers.get('X-User-ID', 'anonymous')
    session_id = create_session(user_id)

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

        gemini_client = gemini_ai.Client(api_key=GEMINI_API_KEY)

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        text_nodes = request.form.get('text_nodes', '').strip()
        text_nodes_block = (
            f"\n\nACCESSIBILITY TEXT NODES (top to bottom):\n{text_nodes}\n"
            if text_nodes else ""
        )

        t0 = int(time.time() * 1000)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                GEMINI_PROMPT + text_nodes_block
            ]
        )
        elapsed_ms = int(time.time() * 1000) - t0

        parsed = parse_gemini_json(response.text or "")
        result = str(parsed.get("result", "UNCLEAR_GARMENT")).strip().upper()
        if result not in {"NO_GARMENT", "UNCLEAR_GARMENT", "PARTIAL_GARMENT", "READY"}:
            result = "UNCLEAR_GARMENT"

        product_title = str(parsed.get("product_title", "") or "").strip()
        brand_name    = str(parsed.get("brand_name", "") or "").strip()
        garment_class = str(parsed.get("garment_class", "") or "").strip()
        if garment_class not in VALID_GARMENT_CLASSES:
            garment_class = ""

        logger.info(f"[{request_id}] Gemini → result={result} title={product_title!r} "
                    f"brand={brand_name!r} class={garment_class!r}")

        usage = response.usage_metadata
        tokens_in  = getattr(usage, 'prompt_token_count',     0) or 0
        tokens_out = getattr(usage, 'candidates_token_count', 0) or 0

        update_validate(session_id, result, elapsed_ms, tokens_in, tokens_out,
                         product_title, brand_name, garment_class)

        return jsonify({
            'result': result,
            'request_id': request_id,
            'session_id': session_id,
            'product_title': product_title,
            'product_brand': brand_name,
            'garment_class': garment_class
        }), 200

    except Exception as e:
        logger.error(f"Validate error: {e}")
        return jsonify({'result': 'ERROR', 'message': str(e)}), 500

    finally:
        cleanup_files([image_path])


# ─────────────────────────────────────────────────────────────
#  /try-on  — starts async processing
# ─────────────────────────────────────────────────────────────

@app.route('/try-on', methods=['POST'])
@require_ai_client
def virtual_try_on():
    person_path  = None
    clothing_path = None
    request_id   = str(uuid.uuid4())
    user_id      = request.headers.get('X-User-ID', 'anonymous')

    
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'error': 'Missing session_id', 
                        'message': 'session_id from /validate is required'}), 400
    try:
        if 'person_image' not in request.files or 'clothing_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Missing required files',
                'message': 'Both person_image and clothing_image are required'
            }), 400

        person_image   = request.files['person_image']
        clothing_image = request.files['clothing_image']
        garment_description = request.form.get('garment_description', 'stylish clothing')
        category            = request.form.get('category', 'upper_body')
        garment_class = request.form.get('garment_class', '').strip()

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

        person_path   = save_uploaded_file(person_image)
        clothing_path = save_uploaded_file(clothing_image)

        if not person_path or not clothing_path:
            return jsonify({'success': False, 'error': 'File processing error',
                            'message': 'Failed to process uploaded files'}), 500

        logger.info(f"[{request_id}] Processing try-on: category={category}, session={session_id}")

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
            category,
            session_id,
            user_id,
            garment_class 
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
#  /try-on/status
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
        return jsonify(result), 200

    if result.get('status') == 'failed':
        with processing_lock:
            processing_results.pop(request_id, None)
        return jsonify(result), 500

    return jsonify({'success': False, 'error': 'Unknown status'}), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json(silent=True) or {}
        product_brand = data.get('product_brand', '')
        product_name  = data.get('product_name', '')
        decision      = data.get('decision', '')
        reason        = data.get('reason', '')

        if not decision:
            return jsonify({'success': False, 'message': 'decision is required'}), 400

        save_feedback(product_brand, product_name, decision, reason)

        return jsonify({'success': True, 'message': 'Feedback saved'}), 200
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    if not debug:
        logger.info("Starting production server...")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)