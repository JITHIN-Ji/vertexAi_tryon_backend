import os
import time
import uuid
from functools import wraps

from flask import request, jsonify

from app.config import (
    app, logger, executor, processing_results, processing_lock,
    VALID_CATEGORIES, VALID_GARMENT_CLASSES, GEMINI_API_KEY,
)
from werkzeug.exceptions import RequestEntityTooLarge

from app.ai_clients import client
from app.gemini_validator import GEMINI_PROMPT, parse_gemini_json
from app.image_utils import validate_image_file, save_uploaded_file, cleanup_files
from app.tryon_processor import process_try_on_background
from app.analytics import create_session, update_validate, save_feedback, get_analytics

from google import genai as gemini_ai
from google.genai import types as genai_types


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
    device_model = request.headers.get('X-Device-Model', '')
    device_manufacturer = request.headers.get('X-Device-Manufacturer', '')
    session_id = create_session(user_id, device_model, device_manufacturer)

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
        reason        = str(parsed.get("reason", "") or "").strip()
        if garment_class not in VALID_GARMENT_CLASSES:
            garment_class = ""

        logger.info(f"[{request_id}] Gemini → result={result} title={product_title!r} "
                    f"brand={brand_name!r} class={garment_class!r} reason={reason!r}")

        usage = response.usage_metadata
        tokens_in  = getattr(usage, 'prompt_token_count',     0) or 0
        tokens_out = getattr(usage, 'candidates_token_count', 0) or 0

        update_validate(session_id, result, elapsed_ms, tokens_in, tokens_out,
                         product_title, brand_name, garment_class,
                         no_garment_reason=reason)

        return jsonify({
            'result': result,
            'request_id': request_id,
            'session_id': session_id,
            'product_title': product_title,
            'product_brand': brand_name,
            'garment_class': garment_class,
            'reason': reason,
            'validate_time_ms': elapsed_ms
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
        validate_time_ms = request.form.get('validate_time_ms', type=int) or 0

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
            garment_class,
            validate_time_ms
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


@app.route('/analytics', methods=['GET'])
def get_analytics_route():
    try:
        search = request.args.get('search', '')
        rows = get_analytics(search)
        return jsonify({'success': True, 'count': len(rows), 'data': rows}), 200
    except Exception as e:
        logger.error(f"Analytics fetch error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
 
@app.route('/analytics-view')
def analytics_view():
    return app.send_static_file('analytics_dashboard.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    if not debug:
        logger.info("Starting production server...")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)