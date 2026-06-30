import os
import json

from google import genai
from google.oauth2 import service_account

from app.config import PROJECT_ID, LOCATION, logger

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