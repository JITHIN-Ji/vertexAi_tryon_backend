import typing
import time

from PIL import Image as PIL_Image
from google.genai.types import (
    Image,
    ProductImage,
    RecontextImageConfig,
    RecontextImageSource,
)

from app.config import logger, executor, processing_results, processing_lock
from app.ai_clients import client
from app.image_utils import prepare_garment_image, pil_image_to_base64, cleanup_files
from app.storage import save_tryon_images_to_supabase
from app.analytics import update_tryon


def process_try_on_background(request_id, person_path, clothing_path,
                               garment_description, category,
                               session_id, user_id, garment_class="",
                               validate_time_ms=0):
    cropped_path = None
    deepfashion_time_ms = 0
    try:
        logger.info(f"[{request_id}] garment_class hint from validate: {garment_class!r}")

        deepfashion_start = int(time.time() * 1000)
        cropped_path = prepare_garment_image(request_id, clothing_path, garment_class)
        deepfashion_time_ms = int(time.time() * 1000) - deepfashion_start

        tryon_start = int(time.time() * 1000)
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
        tryon_elapsed_ms = int(time.time() * 1000) - tryon_start

        logger.info(f"[{request_id}] Google AI API call successful!")

        if not response.generated_images:
            total_time_ms = (validate_time_ms or 0) + deepfashion_time_ms + tryon_elapsed_ms
            step_times = {
                "validate": validate_time_ms or 0,
                "deepfashion": deepfashion_time_ms,
                "tryon": tryon_elapsed_ms,
            }
            slowest_step = max(step_times, key=step_times.get)
            update_tryon(session_id, "failed", tryon_elapsed_ms, "No image generated",
                         deepfashion_time_ms=deepfashion_time_ms,
                         total_time_ms=total_time_ms,
                         slowest_step=slowest_step)
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

        total_time_ms = (validate_time_ms or 0) + deepfashion_time_ms + tryon_elapsed_ms
        step_times = {
            "validate": validate_time_ms or 0,
            "deepfashion": deepfashion_time_ms,
            "tryon": tryon_elapsed_ms,
        }
        slowest_step = max(step_times, key=step_times.get)

        garment_image_path = f"tryon-results/{user_id}/{session_id}/garment.jpg"
        result_image_path = f"tryon-results/{user_id}/{session_id}/result.png"

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
            update_tryon(session_id, "success", tryon_elapsed_ms,
                         garment_image_path=garment_image_path,
                         result_image_path=result_image_path,
                         deepfashion_time_ms=deepfashion_time_ms,
                         total_time_ms=total_time_ms,
                         slowest_step=slowest_step)

        executor.submit(save_in_background)

        logger.info(f"[{request_id}] Processing completed successfully")

    except Exception as e:
        # If we failed before tryon_start was set, fall back to 0 for tryon timing
        tryon_elapsed_ms = locals().get('tryon_elapsed_ms', 0)
        total_time_ms = (validate_time_ms or 0) + deepfashion_time_ms + tryon_elapsed_ms
        step_times = {
            "validate": validate_time_ms or 0,
            "deepfashion": deepfashion_time_ms,
            "tryon": tryon_elapsed_ms,
        }
        slowest_step = max(step_times, key=step_times.get)
        update_tryon(session_id, "failed", tryon_elapsed_ms, str(e),
                     deepfashion_time_ms=deepfashion_time_ms,
                     total_time_ms=total_time_ms,
                     slowest_step=slowest_step)
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