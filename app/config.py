import os
import tempfile
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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

SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "amazon_screenshot_garmentimages")