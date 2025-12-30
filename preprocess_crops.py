import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==========================
# CONFIG
# ==========================
RAW_DIR = "dataset/raw_images"
OUT_DIR = "dataset/processed"
YOLO_MODEL = "yolov8n-seg.pt"

TARGET_CLASS = "banana"
MIN_AREA_RATIO = 0.02   # ignore tiny masks
IMG_SIZE = 224          # for MobileNet

# ==========================
# LOAD YOLO SEGMENTATION
# ==========================
model = YOLO(YOLO_MODEL)

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# HELPER FUNCTIONS
# ==========================
def apply_mask_and_crop(image, mask):
    """
    Apply segmentation mask and crop tight bounding box
    """
    mask = (mask * 255).astype(np.uint8)

    # Morphology to clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find largest contour ONLY
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Crop original image
    cropped = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # Remove background
    bg_removed = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)

    return bg_removed


# ==========================
# MAIN LOOP
# ==========================
for img_path in Path(RAW_DIR).glob("*.*"):
    print(f"Processing {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    results = model(img, verbose=False)[0]

    if results.masks is None:
        print("  ❌ No masks detected")
        continue

    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.cls.cpu().numpy()

    for i, mask in enumerate(masks):
        class_id = int(boxes[i])
        class_name = model.names[class_id]

        # ONLY banana
        if class_name.lower() != TARGET_CLASS:
            continue

        # Ignore very small masks
        if np.sum(mask) / mask.size < MIN_AREA_RATIO:
            continue

        banana_crop = apply_mask_and_crop(img, mask)
        if banana_crop is None:
            continue

        banana_crop = cv2.resize(banana_crop, (IMG_SIZE, IMG_SIZE))

        # 🔴 CHANGE THIS MANUALLY PER FOLDER
        label = "disease"   # healthy / rotten / disease

        save_dir = os.path.join(OUT_DIR, label)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, img_path.stem + ".jpg")
        cv2.imwrite(save_path, banana_crop)

        print(f"  ✅ Saved: {save_path}")

print("🎉 Preprocessing complete!")
