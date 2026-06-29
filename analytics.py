# analytics.py
import os
import uuid
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def _headers():
    return {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json"
    }


def create_session(user_id: str) -> str:
    session_id = str(uuid.uuid4())
    try:
        data = {
            "session_id": session_id,
            "user_id": user_id,
        }
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/tryon_analytics",
            headers=_headers(),
            json=data,
            timeout=5
        )
        if response.status_code in (200, 201):
            logger.info(f"Session created: {session_id}")
        else:
            logger.warning(f"Session create failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"create_session error: {e}")
    return session_id


def update_validate(session_id: str, result: str, time_ms: int,
                    tokens_in: int, tokens_out: int,
                    product_title: str = None, product_brand: str = None,
                    garment_class: str = None):
    try:
        data = {
            "validate_result":     result,
            "validate_time_ms":    time_ms,
            "validate_tokens_in":  tokens_in,
            "validate_tokens_out": tokens_out,
            "product_title":       product_title,
            "product_brand":       product_brand,
            "garment_class":       garment_class,
        }
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/tryon_analytics"
            f"?session_id=eq.{session_id}",
            headers=_headers(),
            json=data,
            timeout=5
        )
        logger.warning(f"PATCH response: {response.status_code} | {response.text}")
    except Exception as e:
        logger.warning(f"update_validate error: {e}")

        

def update_tryon(session_id: str, status: str,
                 time_ms: int, error: str = None,
                 garment_image_path: str = None,
                 result_image_path: str = None):
    try:
        data = {
            "tryon_status":       status,
            "tryon_time_ms":      time_ms, 
            "tryon_error":        error,
            "garment_image_path": garment_image_path,
            "result_image_path":  result_image_path,
        }
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/tryon_analytics"
            f"?session_id=eq.{session_id}",
            headers=_headers(),
            json=data,
            timeout=5
        )
        if response.status_code not in (200, 204):
            logger.warning(f"update_tryon failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"update_tryon error: {e}")


def save_feedback(product_brand: str, product_name: str, decision: str, reason: str):
    try:
        data = {
            "product_brand": product_brand,
            "product_name":  product_name,
            "decision":      decision,
            "reason":        reason,
        }
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/tryon_feedback",
            headers=_headers(),
            json=data,
            timeout=5
        )
        if response.status_code in (200, 201):
            logger.info(f"Feedback saved: {product_brand} / {product_name} / {decision}")
        else:
            logger.warning(f"Feedback save failed: {response.status_code} | {response.text}")
    except Exception as e:
        logger.warning(f"save_feedback error: {e}")