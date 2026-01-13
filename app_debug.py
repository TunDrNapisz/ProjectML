from flask import Flask, request, jsonify, send_from_directory, render_template
import cv2
import numpy as np
import os
import uuid
from groq import Groq
import json  # Tambahan: Untuk format JSON di terminal

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
    print(f"[WARNING] YOLO Model failed: {e}")

try:
    banana_classifier = load_model("efficientnetB0_fruit_model_lt.keras")
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
# NATURAL CROPPING LOGIC
# ===============================
def get_natural_crop(image, results, box_index=0):
    try:
        # Ambil koordinat asal
        coords = results[0].boxes[box_index].xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, coords)

        # 1. TAMBAH PADDING (Ruang ekstra 15% supaya pisang tak nampak terhimpit)
        h, w, _ = image.shape
        pad_w = int((x2 - x1) * 0.15)
        pad_h = int((y2 - y1) * 0.15)

        # Pastikan koordinat baru tidak terkeluar dari saiz imej asal
        nx1 = max(0, x1 - pad_w)
        ny1 = max(0, y1 - pad_h)
        nx2 = min(w, x2 + pad_w)
        ny2 = min(h, y2 + pad_h)

        # 2. PILIHAN: Gunakan pemotongan asal tanpa Masking yang kasar
        # Ini akan memaparkan pisang dengan lebih jelas beserta sedikit latar belakangnya
        crop = image[ny1:ny2, nx1:nx2]

        # Jika anda masih mahu latar belakang putih, gunakan final_img dari logik asal
        # tetapi potong menggunakan koordinat 'n' (nx1, ny1, nx2, ny2)
        if results[0].masks is not None:
            mask = results[0].masks.data[box_index].cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            white_bg = np.ones_like(image) * 255
            masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
            inv_mask = cv2.bitwise_not(binary_mask)
            bg_part = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
            final_img = cv2.add(masked_image, bg_part)

            # Potong imej yang sudah berlatar belakang putih dengan padding
            crop = final_img[ny1:ny2, nx1:nx2]

        return crop
    except Exception as e:
        print(f"Natural crop error: {e}")
        return None


def generate_ai_description(label, scores, ripeness_status="Optimal"):


    try:
        prompt = (
            f"Analyze this banana: {label}. "
            f"Current Data Scores: Healthy {scores['Healthy']}%, Disease {scores['Disease']}%, Rotten {scores['Rotten']}%. "

            "Expert Logic Table: "
            # --- LOGIK HEALTHY ---
            "- If Healthy is 90.0% OR HIGHER: Advice: 'Safe to eat, exceptionally fresh, and has a firm texture.' "
            "- If Healthy is BETWEEN 70.0% AND 89.9%: 'Safe to eat, sweet, and ideal for immediate consumption.' "
            "- If Healthy is BETWEEN 50.0% AND 69.9%: Advice: 'Safe to eat and highly recommended for making cakes or cekodok.' "
            "- If Healthy is BETWEEN 0.0% AND 49.9%: Advice: 'Potentially safe but MUST be inspected manually for hidden decay before eating.' "

            # --- LOGIK DISEASE (DIBETULKAN) ---
            "- If Disease is BETWEEN 0.0% AND 49.9%: Advice: 'Risky but potentially usable for cooking (e.g., pisang goreng) ONLY after deep frying to kill pathogens, but proceed with caution.' "
            "- If Disease is BETWEEN 50.0% AND 69.9%: Advice: 'Strictly unfit for consumption and should be discarded to avoid health risks.' "
            "- If Disease is 70.0% OR HIGHER: Advice: 'DO NOT EAT and discard immediately due to severe contamination.' "

            # --- LOGIK ROTTEN ---
            "- If Rotten is BETWEEN 0.0% AND 49.9%: Advice: 'Unfit for consumption even at low scores, should be discarded to avoid ingestion of mold or toxins.' "
            "- If Rotten is BETWEEN 50.0% AND 69.9%: Advice: 'Unfit for raw consumption and should be discarded due to early decay.' "
            "- If Rotten is 70.0% OR HIGHER: Advice: 'DO NOT EAT and discard immediately due to total decay.' "

            # --- TAMBAHAN UNTUK EXPIRY DATE DINAMIK ---
            "CRITICAL INSTRUCTION FOR EXPIRY: Do not use a fixed value for expiry. "
            "Based on the scores above, use your scientific knowledge to estimate a realistic expiry period. "
            "For example, a 95% healthy banana might last 8 days, while 75% might last 4 days. "
            "If Disease/Rotten is dominant, set expiry to 'Expired' or 'Cook Today'. "

            "CRITICAL: Based on the scores provided, identify which SINGLE rule applies for the advice, "
            "but generate the 'expiry' dynamically. Return ONLY a JSON object: "
            "{'advice': '...', 'expiry': 'your dynamic estimation', 'freshness': 0-100}."
        )

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            # Temperature 0.5 supaya AI boleh buat anggaran hari yang berbeza-beza (dinamik)
            temperature=0.5,
            timeout=5
        )

        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq Error: {e}")
        return None

# ===============================
# STRICT EATABILITY LOGIC (UPDATED)
# ===============================
def get_banana_info(label, accuracy, all_scores, crop_img):
    h_score = round(float(all_scores.get("Healthy", 0.0)), 2)
    d_score = round(float(all_scores.get("Disease", 0.0)), 2)
    r_score = round(float(all_scores.get("Rotten", 0.0)), 2)

    max_score = max(h_score, d_score, r_score)

    # --- LANGKAH 1: TENTUKAN CATEGORY DULU DI PYTHON ---
    category = "Healthy"
    if r_score == max_score:
        category = "Rotten"
    elif d_score == max_score:
        category = "Disease"

    # Ambil data daripada Groq API
    ai_data = generate_ai_description(label, all_scores, category)

    if not ai_data:
        ai_data = {"advice": "Manual inspection required.", "expiry": "N/A", "freshness": 0}

    display_desc = ai_data.get("advice")
    display_expiry = ai_data.get("expiry")
    display_freshness = round(float(ai_data.get("freshness", h_score)), 2)

    # ============================================================
    # 1. LOGIK ROTTEN (Mesti Dominan)
    # ============================================================
    if r_score == max_score and r_score > 0:
        if r_score >= 70.0:
            eat = "NOT SAFE - Discard Immediately"
        elif r_score >= 50.0:
            eat = "Unsafe (Discard)"
        else:  # Bawah 50%
            eat = "Unsafe (Early Decay)"

        return {
            "type": "Rotten Banana",
            "description": display_desc,
            "eatability": eat,
            "expiry": "Expired",
            "freshness": display_freshness,
            "reason": [f"Rotten is dominant ({r_score}%)"]
        }

    # ============================================================
    # 2. LOGIK DISEASE (Mesti Dominan)
    # ============================================================
    elif d_score == max_score and d_score > 0:
        if d_score >= 70.0:
            return {
                "type": "Disease Banana",
                "description": display_desc,
                "eatability": "NOT SAFE - Discard Immediately",  # Ikut Prompt Logik 7
                "expiry": "Expired",
                "freshness": display_freshness,
                "reason": [f"High Disease Score ({d_score}%)"]
            }
        elif d_score >= 50.0:
            return {
                "type": "Uncertain (Disease)",
                "description": display_desc,
                "eatability": "Strictly Unfit for Consumption",  # Ikut Prompt Logik 6
                "expiry": display_expiry,
                "freshness": display_freshness,
                "reason": [f"Moderate Disease Score ({d_score}%)"]
            }
        else:  # Bawah 50%
            return {
                "type": "Uncertain (Potential Disease)",
                "description": display_desc,
                "eatability": "Risky (For Cooking Only)",  # Ikut Prompt Logik 5
                "expiry": display_expiry,
                "freshness": display_freshness,
                "reason": [f"Low Disease Score ({d_score}%)"]
            }

    # ============================================================
    # 3. LOGIK HEALTHY (Mesti Dominan)
    # ============================================================
    elif h_score == max_score and h_score > 0:
        if h_score >= 70.0:
            eat = "Safe to Eat"
            typ = "Healthy Banana"
        elif h_score >= 50.0:
            eat = "Safe (Ideal for Cooking)"  # Ikut Prompt Logik 3
            typ = "Safe"
        else:  # Bawah 50%
            eat = "Manual Inspection Required"  # Ikut Prompt Logik 4
            typ = "Inconclusive (Healthy)"

        return {
            "type": typ,
            "description": display_desc,
            "eatability": eat,
            "expiry": display_expiry,
            "freshness": h_score,
            "reason": [f"Healthy Score ({h_score}%)"]
        }

    # ============================================================
    # 4. FALLBACK
    # ============================================================
    return {
        "type": "Inconclusive",
        "description": "Inconclusive results. Please perform a manual visual inspection.",
        "eatability": "Manual Inspection Required",
        "expiry": display_expiry,
        "freshness": 0,
        "reason": ["Confidence below threshold"]
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
    results = yolo_seg_model(img, conf=0.35, iou=0.45, classes=[46])

    if results and len(results[0].boxes) > 0:
        def calc_area(box):
            c = box.xyxy.cpu().numpy()[0]
            return (c[2] - c[0]) * (c[3] - c[1])

        sorted_indices = np.argsort([calc_area(box) for box in results[0].boxes])[::-1]

        for i in sorted_indices[:4]:
            banana_crop = get_natural_crop(img, results, box_index=i)
            if banana_crop is not None and banana_crop.size > 0:
                class_id, confidence, all_scores = classify_banana(banana_crop)
                if confidence < 0.30: continue

                banana_type = label_map.get(class_id, "Unknown")
                info = get_banana_info(banana_type, confidence, all_scores, banana_crop)
                crop_name = f"crop_{uuid.uuid4()}.jpg"
                cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), banana_crop)

                detections.append({
                    "id": str(uuid.uuid4())[:8],
                    "type": info.get("type", banana_type),  # UBAH INI: Guna info["type"] jika ada
                    "confidence": confidence,  # TAMBAH/KEKALKAN: Untuk index.html baca confidence
                    "accuracy_val": round(confidence * 100, 2),
                    "all_scores": all_scores,
                    "description": info["description"],
                    "eatability": info["eatability"],
                    "expiry": info["expiry"],
                    "freshness": info["freshness"],
                    "crop_url": f"{request.host_url}crop/{crop_name}"
                })

    if not detections:
        class_id, confidence, all_scores = classify_banana(img)
        if confidence >= 0.50:
            banana_type = label_map.get(class_id, "Unknown")
            info = get_banana_info(banana_type, confidence, all_scores, img)
            crop_name = f"fallback_{uuid.uuid4()}.jpg"
            cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), img)

            detections.append({
                "id": "full-scan",
                "type": info.get("type", banana_type),  # UBAH INI JUGA
                "accuracy_val": round(confidence * 100, 2),
                "all_scores": all_scores,
                "description": info["description"] + " (Full Scan)",
                "eatability": info["eatability"],
                "expiry": info["expiry"],
                "freshness": info["freshness"],
                "crop_url": f"{request.host_url}crop/{crop_name}"
            })

    res_plotted = results[0].plot()
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), res_plotted)

    # FINAL JSON RESPONSE (With Responsible AI Disclaimer)
    final_response = {
        "processed_image": f"{request.host_url}output/{filename}",
        "detections": detections,
        "disclaimer": "This result is AI-generated. Please verify with a food safety expert before consumption."
    }

    # Print ke terminal untuk bukti JSON
    import json
    print("\n" + "=" * 50)
    print(json.dumps(final_response, indent=4))
    print("=" * 50)

    return jsonify(final_response)


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