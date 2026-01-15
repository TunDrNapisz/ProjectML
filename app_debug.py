from flask import Flask, request, jsonify, send_from_directory, render_template
import cv2
import numpy as np
import os
import uuid
from groq import Groq
import json  # Tambahan: Untuk format JSON di terminal
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env file
load_dotenv()

from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import Model

# ===============================
# API CLIENTS
# ===============================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "banana-images")

# ===============================
# CONFIGURATION
# ===============================
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
CROP_FOLDER = "crops"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, CROP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__, template_folder="templates")

# Verify Supabase connection
if SUPABASE_URL and SUPABASE_KEY:
    print("[INFO] Supabase client initialized successfully")
else:
    print("[WARNING] Supabase credentials missing - database features will not work")

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
    print("[INFO] EfficientNetB0 classifier loaded successfully")
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
# GRAD-CAM HEATMAP LOGIC
# ===============================
def generate_gradcam_heatmap(model, img_array, class_id):
    """
    Generate Guided Grad-CAM heatmap for the given image and class
    Args:
        model: Keras model
        img_array: Preprocessed image array (1, 224, 224, 3)
        class_id: Target class index
    Returns:
        heatmap: Grad-CAM heatmap (224, 224)
    """
    try:
        # Find the last convolutional layer
        last_conv_layer = None
        last_conv_layer_name = None
        for layer in model.layers[::-1]:
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                last_conv_layer_name = layer.name
                break
        
        if last_conv_layer is None:
            print("[WARNING] No convolutional layer found for Grad-CAM")
            return None
        
        print(f"[DEBUG] Using conv layer: {last_conv_layer_name}")
        
        # Create a model that outputs both the predictions and the conv layer outputs
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.output, last_conv_layer.output]
        )
        
        # Convert input to tensor
        img_tensor = tf.cast(img_array, tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            # Get predictions and conv outputs
            model_outputs = grad_model(img_tensor, training=False)
            
            # Handle nested list structure from Keras model
            if isinstance(model_outputs, list):
                # model_outputs = [[predictions], conv_outputs]
                predictions_raw = model_outputs[0]
                conv_outputs = model_outputs[1]
                
                # Extract actual predictions tensor
                if isinstance(predictions_raw, list):
                    predictions = predictions_raw[0]
                else:
                    predictions = predictions_raw
            else:
                predictions, conv_outputs = model_outputs
            
            class_channel = predictions[:, class_id]
        
        print(f"[DEBUG] Predictions shape: {predictions.shape}, class_id: {class_id}, prediction value: {predictions[0, class_id].numpy():.4f}")
        
        # Get gradients of the class with respect to conv outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            print("[WARNING] Gradients are None")
            return None
        
        print(f"[DEBUG] Gradients shape: {grads.shape}, min: {tf.math.reduce_min(grads).numpy():.4f}, max: {tf.math.reduce_max(grads).numpy():.4f}")
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(f"[DEBUG] Pooled gradients shape: {pooled_grads.shape}")
        
        # Multiply each channel by its gradient weight
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Guided Grad-CAM: Apply ReLU to both gradients and activations
        # This amplifies positive contributions (where the model focuses)
        guided_grads = tf.cast(grads > 0, tf.float32) * grads
        guided_pooled_grads = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
        heatmap_guided = conv_outputs_val @ guided_pooled_grads[..., tf.newaxis]
        heatmap_guided = tf.squeeze(heatmap_guided)
        
        # Normalize the heatmap
        heatmap_combined = heatmap + heatmap_guided  # Combine both for better results
        heatmap_combined = tf.maximum(heatmap_combined, 0)
        heatmap_max = tf.math.reduce_max(heatmap_combined)
        
        if heatmap_max > 0:
            heatmap_combined = heatmap_combined / heatmap_max
        
        heatmap_np = heatmap_combined.numpy()
        print(f"[DEBUG] Final heatmap - shape: {heatmap_np.shape}, range: [{heatmap_np.min():.3f}, {heatmap_np.max():.3f}]")
        
        return heatmap_np
        
    except Exception as e:
        print(f"[ERROR] Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def overlay_gradcam_on_image(img_rgb, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the original image
    Args:
        img_rgb: Original image in RGB (224, 224, 3)
        heatmap: Grad-CAM heatmap (224, 224)
        alpha: Transparency of heatmap overlay
    Returns:
        overlayed_img: Image with heatmap overlay (BGR format for OpenCV)
    """
    try:
        if heatmap is None:
            print("[WARNING] Heatmap is None, returning original image")
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        print(f"[DEBUG] Overlay: img_rgb shape: {img_rgb.shape}, heatmap shape: {heatmap.shape}")
        
        # Ensure heatmap is 2D
        if len(heatmap.shape) > 2:
            heatmap = heatmap[:, :, 0]
        
        # Resize heatmap to match image size
        h, w = img_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        print(f"[DEBUG] Heatmap resized to: {heatmap_resized.shape}")
        
        # Normalize heatmap to 0-255 range
        heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap (JET for better visualization - red hot areas)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        print(f"[DEBUG] Heatmap colored shape: {heatmap_colored.shape}")
        
        # Convert img_rgb to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Blend the images
        overlayed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        print(f"[DEBUG] Overlay completed successfully")
        return overlayed
    except Exception as e:
        print(f"[ERROR] Heatmap overlay failed: {e}")
        import traceback
        traceback.print_exc()
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


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
        print(f"[DEBUG] Generating AI description for: {label}")
        print(f"[DEBUG] Scores: {scores}")
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

        print("[DEBUG] Calling Groq API...")
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            # Temperature 0.5 supaya AI boleh buat anggaran hari yang berbeza-beza (dinamik)
            temperature=0.5,
            timeout=5
        )

        response_json = json.loads(completion.choices[0].message.content)
        print(f"[DEBUG] Groq response: {response_json}")
        return response_json
    except Exception as e:
        print(f"[ERROR] Groq Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===============================
# STRICT EDIBLE LOGIC FUNCTION
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
                "type": "Uncertain",
                "description": display_desc,
                "eatability": "Advised against Consumption",  # Ikut Prompt Logik 6
                "expiry": display_expiry,
                "freshness": display_freshness,
                "reason": [f"Moderate Disease Score ({d_score}%)"]
            }
        else:  # Bawah 50%
            return {
                "type": "Uncertain",
                "description": display_desc,
                "eatability": "Not recommended for raw consumption (For Cooking Only)",  # Ikut Prompt Logik 5
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
        if banana_classifier is None:
            print("[ERROR] Banana classifier model is not loaded!")
            return 1, 0.0, {"Disease": 0, "Healthy": 0, "Rotten": 0}, None
        
        print("[DEBUG] Starting banana classification...")
        processed_img = apply_preprocessing(img_bgr)
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = img_to_array(img_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        print("[DEBUG] Running model prediction...")
        preds = banana_classifier.predict(img_array, verbose=0)[0]
        print(f"[DEBUG] Raw predictions: {preds}")
        
        all_scores = {
            "Disease": float(preds[0]) * 100,
            "Healthy": float(preds[1]) * 100,
            "Rotten": float(preds[2]) * 100
        }
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        print(f"[DEBUG] Class ID: {class_id}, Confidence: {confidence}")
        print(f"[DEBUG] All scores: {all_scores}")
        
        # Generate Grad-CAM heatmap
        print("[DEBUG] Generating Grad-CAM heatmap...")
        heatmap = generate_gradcam_heatmap(banana_classifier, img_array, class_id)
        
        # Overlay heatmap on ORIGINAL crop (preserves aspect ratio)
        heatmap_img = None
        if heatmap is not None:
            print(f"[DEBUG] Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
            # Use original img_rgb (processed but not resized to 224x224) for overlay
            heatmap_img = overlay_gradcam_on_image(img_rgb, heatmap, alpha=0.5)
            if heatmap_img is not None:
                print(f"[DEBUG] Heatmap overlay successful, shape: {heatmap_img.shape}")
            else:
                print("[WARNING] Heatmap overlay returned None")
        else:
            print("[WARNING] Grad-CAM heatmap generation returned None")
        
        return class_id, confidence, all_scores, heatmap_img
    except Exception as e:
        print(f"[ERROR] Classification error: {e}")
        import traceback
        traceback.print_exc()
        return 1, 0.0, {"Disease": 0, "Healthy": 0, "Rotten": 0}, None


# ===============================
# SUPABASE UPLOAD & DATABASE
# ===============================
def upload_to_supabase(filepath, filename):
    """Upload image to Supabase storage and return public URL"""
    try:
        if not (SUPABASE_URL and SUPABASE_KEY):
            print("[WARNING] Supabase not configured, skipping upload")
            return None
        
        with open(filepath, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                filename,
                f,
                {"content-type": "image/jpeg"}
            )
        
        url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        print(f"[INFO] Image uploaded to Supabase: {filename}")
        return url
    except Exception as e:
        print(f"[WARNING] Supabase upload failed: {e}")
        return None


def save_prediction_to_db(image_url, prediction, confidence):
    """Save prediction result to Supabase database"""
    try:
        if not (SUPABASE_URL and SUPABASE_KEY):
            print("[WARNING] Supabase not configured, skipping database save")
            return False
        
        supabase.table("banana_predictions").insert({
            "image_url": image_url,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        }).execute()
        
        print("[INFO] Prediction saved to database")
        return True
    except Exception as e:
        print(f"[WARNING] Database save failed: {e}")
        return False


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
                class_id, confidence, all_scores, heatmap_img = classify_banana(banana_crop)
                if confidence < 0.30: continue

                banana_type = label_map.get(class_id, "Unknown")
                info = get_banana_info(banana_type, confidence, all_scores, banana_crop)
                crop_name = f"crop_{uuid.uuid4()}.jpg"
                cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), banana_crop)

                # Save heatmap if generated
                heatmap_name = None
                if heatmap_img is not None:
                    heatmap_name = f"heatmap_{uuid.uuid4()}.jpg"
                    cv2.imwrite(os.path.join(CROP_FOLDER, heatmap_name), heatmap_img)
                    print(f"[INFO] Heatmap saved: {heatmap_name}")

                # Create annotated image with confidence display
                annotated_crop = banana_crop.copy()
                cv2.putText(annotated_crop, f"Confidence: {confidence*100:.1f}%", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(annotated_crop, f"Type: {banana_type}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                annotated_crop_name = f"annotated_{uuid.uuid4()}.jpg"
                cv2.imwrite(os.path.join(CROP_FOLDER, annotated_crop_name), annotated_crop)

                # Upload to Supabase and save to database
                image_url = upload_to_supabase(filepath, filename)
                if image_url:
                    save_prediction_to_db(image_url, banana_type, confidence)

                detection_obj = {
                    "id": str(uuid.uuid4())[:8],
                    "type": info.get("type", banana_type),  # UBAH INI: Guna info["type"] jika ada
                    "confidence": confidence,  # TAMBAH/KEKALKAN: Untuk index.html baca confidence
                    "confidence_percentage": round(confidence * 100, 2),
                    "confidence_level": "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low",
                    "accuracy_val": round(confidence * 100, 2),
                    "all_scores": all_scores,
                    "description": info["description"],
                    "eatability": info["eatability"],
                    "expiry": info["expiry"],
                    "freshness": info["freshness"],
                    "crop_url": f"{request.host_url}crop/{crop_name}",
                    "annotated_crop_url": f"{request.host_url}crop/{annotated_crop_name}"
                }
                
                # Add heatmap URL if available
                if heatmap_name:
                    detection_obj["heatmap_url"] = f"{request.host_url}crop/{heatmap_name}"
                
                detections.append(detection_obj)

    if not detections:
        # No bananas detected by YOLO - return error message instead of fallback classification
        res_plotted = results[0].plot()
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), res_plotted)
        
        return jsonify({
            "error": "No banana detected",
            "message": "The uploaded image does not contain any banana. Please upload an image with a clear banana for analysis.",
            "processed_image": f"{request.host_url}output/{filename}",
            "detections": [],
            "disclaimer": {
                "title": "⚠️ IMPORTANT: Responsible AI Notice",
                "main_text": "This result is AI-generated and for informational purposes only.",
                "warning_points": [
                    "Ensure the image clearly shows a banana for accurate detection.",
                    "Try uploading an image with better lighting and clearer view of the banana."
                ]
            }
        }), 400

    res_plotted = results[0].plot()
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), res_plotted)

    # FINAL JSON RESPONSE (With Enhanced Responsible AI Disclaimer)
    final_response = {
        "processed_image": f"{request.host_url}output/{filename}",
        "detections": detections,
        "disclaimer": {
            "title": "⚠️ IMPORTANT: Responsible AI Notice",
            "main_text": "This result is AI-generated and for informational purposes only. It should NOT be used as the sole basis for any decisions.",
            "warning_points": [
                "AI models can make mistakes. Always verify results with a qualified food safety expert.",
                "Check the confidence score - lower confidence (below 70%) means less reliable results.",
                "Review the highlighted regions: Does the AI's detection match what you see?",
                "If you're unsure, discard the banana to avoid food safety risks."
            ],
            "expert_consultation": "For disease diagnosis, consult a food safety expert or agricultural specialist before treatment."
        }
    }

    # Print ke terminal untuk bukti JSON
    import json
    print("\n" + "=" * 50)
    print(json.dumps(final_response, indent=4))
    print("=" * 50)

    return jsonify(final_response)

@app.route("/responsible-ai", methods=["GET"])
def responsible_ai():
    return render_template("responsible-ai.html")

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