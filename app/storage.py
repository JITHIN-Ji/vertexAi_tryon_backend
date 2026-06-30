import base64

from  app.config import logger, SUPABASE_BUCKET
import os


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