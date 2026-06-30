

import os
import cv2
import numpy as np

BG_COLOR_BGR   = (245, 245, 240)   # warm off-white
BG_PADDING     = 40
CONF_THRESHOLD = 0.25


def composite_on_bg(rgba_crop, bg_color_bgr=BG_COLOR_BGR, padding=BG_PADDING):
    """Alpha-composite a BGRA garment crop onto a solid off-white canvas."""
    h, w = rgba_crop.shape[:2]
    canvas_h = h + 2 * padding
    canvas_w = w + 2 * padding

    canvas = np.full((canvas_h, canvas_w, 3), bg_color_bgr, dtype=np.uint8)

    bgr   = rgba_crop[:, :, :3].astype(np.float32)
    alpha = rgba_crop[:, :, 3].astype(np.float32) / 255.0

    roi = canvas[padding:padding + h, padding:padding + w].astype(np.float32)
    blended = alpha[:, :, np.newaxis] * bgr + (1.0 - alpha[:, :, np.newaxis]) * roi
    canvas[padding:padding + h, padding:padding + w] = blended.astype(np.uint8)
    return canvas


def extract_garment_by_class(image_path, model, garment_class=None, conf_threshold=CONF_THRESHOLD):
    """
    Run YOLO segmentation on image_path.

    If garment_class is given and is among the detected classes, isolate
    ONLY that class. Otherwise (class missing/empty/not found) fall back to
    the single largest-area detected garment.

    Returns:
        {
            "garment_image": BGRA np.ndarray or None,
            "class_name":    str or None,
            "bbox":          (x1, y1, x2, y2) or None,
            "used_fallback": bool
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(image_path, conf=conf_threshold)
    result  = results[0]

    if result.masks is None or len(result.masks.data) == 0:
        return {"garment_image": None, "class_name": None, "bbox": None, "used_fallback": False}

    class_names     = model.names
    used_fallback   = False
    selected_cls_id = None

    if garment_class:
        for box in result.boxes:
            cid = int(box.cls.item())
            if class_names[cid] == garment_class:
                selected_cls_id = cid
                break

    if selected_cls_id is None:
        used_fallback = True
        best_area = -1
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area       = area
                selected_cls_id = int(box.cls.item())

    img  = cv2.imread(image_path)
    h, w = img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask_tensor, box in zip(result.masks.data, result.boxes):
        cid = int(box.cls.item())
        if cid != selected_cls_id:
            continue
        mask_np      = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_uint8   = (mask_resized * 255).astype(np.uint8)
        combined_mask = np.maximum(combined_mask, mask_uint8)

    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(combined_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return {"garment_image": None, "class_name": class_names[selected_cls_id],
                "bbox": None, "used_fallback": used_fallback}

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    rgba          = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = combined_mask
    garment_crop  = rgba[y1:y2, x1:x2]

    return {
        "garment_image": garment_crop,
        "class_name":    class_names[selected_cls_id],
        "bbox":          (int(x1), int(y1), int(x2), int(y2)),
        "used_fallback": used_fallback,
    }