from flask import Flask, request, jsonify, send_from_directory, render_template
import cv2
import numpy as np
import os
import uuid

from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===============================
# CONFIGURATION
# ===============================
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
CROP_FOLDER = "crops"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, CROP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__, template_folder="templates")

# ===============================
# LOAD MODELS
# ===============================
yolo_seg_model = None
banana_classifier = None

try:
    yolo_seg_model = YOLO("yolov8s-seg.pt")
    print("[INFO] YOLOv8s (Small) segmentation model loaded")
except Exception as e:
    print(f"[WARNING] Model failed: {e}")

try:
    banana_classifier = load_model("efficientnetB0_fruit_model.h5")
    print("[INFO] EfficientNetB0 classifier loaded")
except Exception as e:
    print(f"[WARNING] EfficientNetB0 classifier failed: {e}")

label_map = {
    0: "Disease Banana",
    1: "Healthy Banana",
    2: "Rotten Banana"
}


# ===============================
# PRE-PROCESSING LOGIC
# ===============================
def apply_preprocessing(img_bgr):
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return img_bgr


# ===============================
# PRECISE CROPPING LOGIC (Old - with Mask)
# ===============================
def get_precise_crop(image, results, box_index=0):
    try:
        if results[0].masks is not None:
            mask = results[0].masks.data[box_index].cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            isolated_banana = image * (mask[:, :, None] > 0.5)
            coords = results[0].boxes[box_index].xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, coords)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            crop = isolated_banana[y1:y2, x1:x2]
            return crop
    except Exception as e:
        print(f"Precise crop error: {e}")
    return None


# ===============================
# NATURAL CROPPING LOGIC (LATEST - Solve Rotten Error)
# ===============================
def get_natural_crop(image, results, box_index=0):
    """
    Memenuhi arahan Dr. Choo: 'Do not remove background'.
    Mengekalkan tekstur asal tanpa mask hitam supaya EfficientNet tidak keliru.
    """
    try:
        coords = results[0].boxes[box_index].xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, coords)

        # Tambah margin 5% supaya imej tidak terlalu rapat (mengurangkan gangguan tepi)
        h, w = image.shape[:2]
        pad_w = int((x2 - x1) * 0.05)
        pad_h = int((y2 - y1) * 0.05)

        nx1, ny1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        nx2, ny2 = min(w, x2 + pad_w), min(h, y2 + pad_h)

        return image[ny1:ny2, nx1:nx2]
    except Exception as e:
        print(f"Natural crop error: {e}")
        return None


# ===============================
# SMART EATABILITY LOGIC (Reflect Reality)
# ===============================
def get_banana_info(label, accuracy, all_scores, crop_img):
    h_score = all_scores.get("Healthy", 0)
    r_score = all_scores.get("Rotten", 0)
    d_score = all_scores.get("Disease", 0)

    # Formula freshness realistik mengikut arahan Dr. Choo
    raw_freshness = (h_score * 1.0 + d_score * 0.7 + r_score * 0.1)
    freshness_val = min(max(raw_freshness, 0), 100)

    if label == "Rotten Banana":
        if r_score > 95.0:
            freshness_val = min(freshness_val, 15.0)
        elif r_score > 80.0:
            freshness_val = max(freshness_val, 60.0)
        else:
            freshness_val = max(freshness_val, 55.0)

    freshness_val = round(freshness_val, 2)

    # Logik PISANG HIJAU (Guna HSV untuk Skin Analysis)
    if label == "Healthy Banana" and crop_img is not None:
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        # Range warna hijau pisang
        mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
        green_ratio = np.count_nonzero(mask) / (crop_img.size / 3)

        if green_ratio > 0.12:  # Threshold untuk pisang mentah
            return {
                "description": "Banana is still green. Perfect for cooking (pisang goreng) or wait to ripen.",
                "eatability": "Wait 2-4 Days",
                "expiry": "5 - 8 Days",
                "freshness": freshness_val
            }

    # Logik PISANG MASAK RANUM (Mencerminkan Realiti - Marketable)
    if label == "Rotten Banana" or r_score > 70.0:
        if freshness_val >= 35.0:
            return {
                "description": "Banana is overripe (masak ranum). Very sweet! Excellent for smoothies or cakes.",
                "eatability": "Safe (Overripe)",
                "expiry": "1 Day",
                "freshness": freshness_val
            }
        else:
            return {
                "description": "Decomposition or mold detected. Toxic risk, do not consume.",
                "eatability": "Not Safe - Discard",
                "expiry": "Expired",
                "freshness": freshness_val
            }

    if label == "Healthy Banana":
        return {
            "description": "Standard ripe banana. Optimal texture and nutrition.",
            "eatability": "Safe to Eat",
            "expiry": "2 - 4 Days",
            "freshness": freshness_val
        }

    # Logik PISANG BERBINTIK (Disease/Spots)
    return {
        "description": "Surface spots detected. Internal fruit is usually safe and sweet.",
        "eatability": "Safe (Peel thoroughly)",
        "expiry": "1 - 2 Days",
        "freshness": freshness_val
    }


# ===============================
# CLASSIFICATION FUNCTION
# ===============================
def classify_banana(img_bgr):
    try:
        processed_img = apply_preprocessing(img_bgr)
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = img_to_array(img_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = banana_classifier.predict(img_array, verbose=0)[0]
        all_scores = {
            "Disease": float(preds[0]) * 100,
            "Healthy": float(preds[1]) * 100,
            "Rotten": float(preds[2]) * 100
        }
        class_id = int(np.argmax(preds))
        return class_id, float(np.max(preds)), all_scores
    except Exception as e:
        print(f"Classification error: {e}")
        return 1, 0.0, {"Disease": 0, "Healthy": 0, "Rotten": 0}


# ===============================
# MAIN DETECT ROUTE
# ===============================
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    detections = []
    # YOLOv8s pengesanan (Visual Analysis)
    results = yolo_seg_model(img, conf=0.20, iou=0.45, classes=[46])

    if results and len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            # 1. FOKUS: Gunakan Natural Crop untuk elak ralat warna busuk
            banana_crop = get_natural_crop(img, results, box_index=i)

            if banana_crop is not None and banana_crop.size > 0:
                # 2. KLASIFIKASI: Analisis kulit pisang
                class_id, confidence, all_scores = classify_banana(banana_crop)

                # Filter gangguan (contoh: tangan/bayang) dengan threshold lebih tinggi
                if confidence < 0.40: continue

                banana_type = label_map.get(class_id, "Unknown")
                info = get_banana_info(banana_type, confidence, all_scores, banana_crop)

                crop_name = f"crop_{uuid.uuid4()}.jpg"
                cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), banana_crop)

                detections.append({
                    "id": str(uuid.uuid4())[:8],
                    "type": banana_type,
                    "accuracy_val": confidence,
                    "all_scores": all_scores,
                    "description": info["description"],
                    "eatability": info["eatability"],
                    "expiry": info["expiry"],
                    "freshness": info["freshness"],
                    "crop_url": f"{request.host_url}crop/{crop_name}"
                })

    # MODIFIKASI: Hanya klasifikasi imej penuh jika YOLO gagal dan imej sangat meyakinkan
    if not detections:
        class_id, confidence, all_scores = classify_banana(img)

        # Logik elak logo universiti/FAIX dikesan sebagai pisang
        if confidence < 0.70:
            return jsonify({
                "processed_image": f"{request.host_url}output/{filename}",
                "detections": [],
                "message": "No banana detected in Visual Analysis. Object ignored to avoid false results."
            })

        banana_type = label_map.get(class_id, "Unknown")
        info = get_banana_info(banana_type, confidence, all_scores, img)
        crop_name = f"fallback_{uuid.uuid4()}.jpg"
        cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), img)

        detections.append({
            "id": "forced",
            "type": banana_type,
            "accuracy_val": confidence,
            "all_scores": all_scores,
            "description": info["description"] + " (Full-scan detection)",
            "eatability": info["eatability"],
            "expiry": info["expiry"],
            "freshness": info["freshness"],
            "crop_url": f"{request.host_url}crop/{crop_name}"
        })

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), results[0].plot())
    return jsonify({
        "processed_image": f"{request.host_url}output/{filename}",
        "detections": detections
    })


# ===============================
# HELPER ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/crop/<filename>")
def crop_file(filename):
    return send_from_directory(CROP_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)