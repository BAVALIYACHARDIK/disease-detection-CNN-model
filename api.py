from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional
import json
import os
 
# Load environment variables from .env if available
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
import httpx
import asyncio
import random

try:
    import cv2
except ImportError:
    cv2 = None

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases from leaf images",
    version="1.0.0"
)

# Add CORS middleware for web/mobile app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
_gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_advice_cache: Dict[str, Dict] = {}
_advice_cache_ttl = int(os.getenv("ADVICE_CACHE_TTL", "3600"))  # seconds

# Small static fallback map when Gemini is unavailable
_static_advice_fallback = {
    "Apple - Apple Scab": (
        "Immediate: remove infected leaves and avoid overhead watering.\n"
        "Cultural: improve air flow and avoid dense planting.\n"
        "Chemical: apply fungicides containing mancozeb or chlorothalonil as labeled.\n"
        "Prevention: rotate crops and prune for ventilation."
    ),
    "Tomato - Early Blight": (
        "Immediate: remove affected foliage and destroy plant debris.\n"
        "Cultural: avoid wetting leaves when irrigating; stake plants.\n"
        "Chemical: consider fungicides with chlorothalonil or copper compounds.\n"
        "Prevention: use resistant varieties and rotate crops."
    )
}
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


def format_disease_name(raw_class_name):
    """
    Format the raw class name into a human-readable format.
    
    Args:
        raw_class_name (str): Raw class name like 'Apple___Apple_scab'
    
    Returns:
        dict: Formatted plant and disease information
    """
    # Split plant and disease
    parts = raw_class_name.split('___')
    if len(parts) != 2:
        return {
            'plant': raw_class_name,
            'disease': 'Unknown',
            'formatted_name': raw_class_name,
            'is_healthy': False
        }
    
    plant_raw, disease_raw = parts
    
    # Format plant name
    plant = plant_raw.replace('_', ' ').replace('(', '').replace(')', '').replace(',', '').strip()
    plant = ' '.join(word.capitalize() for word in plant.split())
    
    # Format disease name
    if disease_raw.lower() == 'healthy':
        disease = 'Healthy'
        formatted_name = f"{plant} - Healthy"
        is_healthy = True
    else:
        # Clean up disease name
        disease = disease_raw.replace('_', ' ').strip()
        
        # Handle special cases
        disease_mappings = {
            'Apple scab': 'Apple Scab',
            'Black rot': 'Black Rot',
            'Cedar apple rust': 'Cedar Apple Rust',
            'Powdery mildew': 'Powdery Mildew',
            'Cercospora leaf spot Gray leaf spot': 'Cercospora Leaf Spot (Gray Leaf Spot)',
            'Common rust': 'Common Rust',
            'Northern Leaf Blight': 'Northern Leaf Blight',
            'Esca (Black Measles)': 'Esca (Black Measles)',
            'Leaf blight (Isariopsis Leaf Spot)': 'Leaf Blight (Isariopsis Leaf Spot)',
            'Haunglongbing (Citrus greening)': 'Huanglongbing (Citrus Greening)',
            'Bacterial spot': 'Bacterial Spot',
            'Early blight': 'Early Blight',
            'Late blight': 'Late Blight',
            'Leaf scorch': 'Leaf Scorch',
            'Leaf Mold': 'Leaf Mold',
            'Septoria leaf spot': 'Septoria Leaf Spot',
            'Spider mites Two-spotted spider mite': 'Spider Mites (Two-spotted Spider Mite)',
            'Target Spot': 'Target Spot',
            'Tomato Yellow Leaf Curl Virus': 'Tomato Yellow Leaf Curl Virus',
            'Tomato mosaic virus': 'Tomato Mosaic Virus'
        }
        
        disease = disease_mappings.get(disease, disease.title())
        formatted_name = f"{plant} - {disease}"
        is_healthy = False
    
    return {
        'plant': plant,
        'disease': disease,
        'formatted_name': formatted_name,
        'is_healthy': is_healthy,
        'raw_class': raw_class_name
    }


def load_model():
    """Load the trained model."""
    global model
    if model is None:
        base_dir = Path(__file__).resolve().parent
        keras_path = base_dir / "trained_model.keras"
        h5_path = base_dir / "trained_model.h5"
        
        if keras_path.exists():
            model_path = keras_path
        elif h5_path.exists():
            model_path = h5_path
        else:
            raise FileNotFoundError("Model file not found. Expected 'trained_model.keras' or 'trained_model.h5'")
        
        model = tf.keras.models.load_model(str(model_path))
    return model


async def get_treatment_advice(plant: str, disease: str) -> Optional[Dict[str, str]]:
    """
    Use Gemini REST API to generate concise, safe treatment advice for a specific plant disease.
    Returns a dict with 'summary'. If API key isn't configured or request fails, returns None.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    cache_key = f"{plant}|||{disease}"
    # Return cached advice if fresh
    cached = _advice_cache.get(cache_key)
    if cached and (asyncio.get_event_loop().time() - cached.get("ts", 0) < _advice_cache_ttl):
        result = {"summary": cached["summary"], "source": "cache"}
        # Include structured data if it was cached
        if "structured" in cached:
            result["structured"] = cached["structured"]
        return result
    # Build a focused prompt; avoid hallucinations and request actionable, region-agnostic guidance.
    prompt = (
        f"You are an expert plant pathologist. Provide treatment advice for {plant} with {disease}.\n"
        "Return ONLY a JSON object (no extra text) with these keys:"
        "\n- short_summary: a 1-2 sentence plain text summary."
        "\n- immediate_steps: an array of short action strings (2-4 items)."
        "\n- cultural_practices: an array of practical cultural/organic controls (2-5 items)."
        "\n- chemical_options: an array of objects with {active_ingredient, notes} for common actives (if applicable)."
        "\n- prevention: an array of prevention tips (2-4 items)."
        "\nKeep each item concise. Avoid brand names, regional legal advice, and more than 200 words total."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{_gemini_model_name}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    attempts = 3
    base_backoff = 1.0
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            timeout = httpx.Timeout(12.0, connect=6.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(url, json=payload, headers=headers)

            status = res.status_code
            if status == 200:
                data = res.json()
                text_parts: List[str] = []
                for cand in (data.get("candidates") or []):
                    content = cand.get("content") or {}
                    for part in (content.get("parts") or []):
                        t = part.get("text")
                        if t:
                            text_parts.append(t)
                summary = ("\n\n".join(text_parts)).strip()
                if summary:
                    # Try to parse strict JSON from the model output
                    structured = None
                    try:
                        structured = json.loads(summary)
                    except Exception:
                        # attempt to extract JSON substring (handle markdown wrapper)
                        import re
                        # First try to remove ```json wrapper if present
                        clean_summary = re.sub(r'^```json\s*\n?', '', summary)
                        clean_summary = re.sub(r'\n?```\s*$', '', clean_summary)
                        try:
                            structured = json.loads(clean_summary)
                        except Exception:
                            # fallback: extract JSON object
                            m = re.search(r"\{[\s\S]*\}", clean_summary)
                            if m:
                                try:
                                    structured = json.loads(m.group(0))
                                except Exception:
                                    structured = None

                    # cache raw summary and structured if available
                    cache_data = {"summary": summary, "ts": asyncio.get_event_loop().time()}
                    result = {"summary": summary, "source": "gemini"}
                    if structured and isinstance(structured, dict):
                        result["structured"] = structured
                        cache_data["structured"] = structured
                    _advice_cache[cache_key] = cache_data
                    return result
                last_error = "Empty response from Gemini"
                break

            # Handle rate limiting
            if status == 429:
                # respect Retry-After if present
                ra = res.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else (base_backoff * (2 ** (attempt - 1)))
                wait += random.uniform(0, 0.5)
                await asyncio.sleep(wait)
                continue

            # Retry on server errors
            if 500 <= status < 600:
                backoff = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                await asyncio.sleep(backoff)
                last_error = f"Server error {status}"
                continue

            # Client errors (bad key, invalid model, etc.) — don't retry
            try:
                err_text = res.text
            except Exception:
                err_text = f"HTTP {status}"
            last_error = f"Client error {status}: {err_text}"
            break

        except httpx.RequestError as e:
            # network error — retry
            last_error = f"Request error: {str(e)}"
            backoff = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            await asyncio.sleep(backoff)
            continue

    # If we get here, Gemini failed. Try static fallback first.
    key = f"{plant} - {disease}"
    fallback = _static_advice_fallback.get(key)
    if fallback:
        return {"summary": fallback, "source": "fallback"}

    # Return structured error so caller can surface it
    return {"error": last_error or "Gemini request failed"}


def preprocess_image(image_bytes: bytes, target_size=(128, 128)):
    """Preprocess image for model prediction."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model_img = pil_img.resize(target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(model_img)
    input_arr = np.expand_dims(input_arr, axis=0)
    return input_arr, pil_img


def analyze_image_quality(pil_img):
    """Comprehensive image quality and leaf validation."""
    np_img = np.asarray(pil_img.resize((224, 224)))
    
    # 1. Green ratio calculation (enhanced)
    r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
    green_mask = (g > r + 10) & (g > b + 10)
    green_ratio = float(np.count_nonzero(green_mask)) / float(np_img.shape[0] * np_img.shape[1])
    
    # 2. Plant-like color detection (improved)
    # Check for natural plant colors (various shades of green, brown, yellow)
    plant_colors = 0
    # Green variations
    light_green = (g > r) & (g > b) & (g > 50)
    plant_colors += np.count_nonzero(light_green)
    # Brown/yellow (dried leaves)
    brown_yellow = (r > 100) & (g > 80) & (b < r * 0.8) & (b < g * 0.8)
    plant_colors += np.count_nonzero(brown_yellow)
    
    plant_color_ratio = float(plant_colors) / float(np_img.shape[0] * np_img.shape[1])
    
    # 3. Color diversity check (leaves have varied but limited color palette)
    unique_colors = len(np.unique(np_img.reshape(-1, 3), axis=0))
    color_diversity = unique_colors / (224 * 224)  # Normalized
    
    # 4. Edge detection (leaves have organic edges)
    edges = 0
    edge_density = 0
    if cv2 is not None:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
        # Blur detection
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Edge detection for organic shapes
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (224 * 224)
    else:
        blur_var = None
    
    # 5. Brightness and contrast check
    brightness = np.mean(np_img)
    contrast = np.std(np_img)
    
    # 6. Unnatural color detection (detect artificial/synthetic images)
    # Very bright artificial colors
    neon_colors = ((r > 200) & (g < 100) & (b < 100)) | \
                  ((r < 100) & (g > 200) & (b < 100)) | \
                  ((r < 100) & (g < 100) & (b > 200)) | \
                  ((r > 200) & (g > 200) & (b < 50)) | \
                  ((r > 200) & (g < 50) & (b > 200)) | \
                  ((r < 50) & (g > 200) & (b > 200))
    neon_ratio = float(np.count_nonzero(neon_colors)) / float(np_img.shape[0] * np_img.shape[1])
    
    # 7. Skin tone detection (to reject human photos)
    skin_mask = ((r > 95) & (g > 40) & (b > 20) & 
                 (r > g) & (r > b) & 
                 (abs(r - g) > 15) & 
                 ((r - g) > 15))
    skin_ratio = float(np.count_nonzero(skin_mask)) / float(np_img.shape[0] * np_img.shape[1])
    
    # 8. Solid background detection (many plant photos have backgrounds)
    # Check corners for solid colors (indicating backgrounds)
    corner_size = 20
    corners = [
        np_img[:corner_size, :corner_size],  # top-left
        np_img[:corner_size, -corner_size:], # top-right
        np_img[-corner_size:, :corner_size], # bottom-left
        np_img[-corner_size:, -corner_size:] # bottom-right
    ]
    
    background_uniformity = 0
    for corner in corners:
        corner_std = np.std(corner)
        if corner_std < 20:  # Low variation indicates solid background
            background_uniformity += 1
    background_score = background_uniformity / 4.0
    
    return {
        'green_ratio': green_ratio,
        'plant_color_ratio': plant_color_ratio,
        'color_diversity': color_diversity,
        'edge_density': edge_density,
        'brightness': brightness,
        'contrast': contrast,
        'neon_ratio': neon_ratio,
        'skin_ratio': skin_ratio,
        'background_score': background_score,
        'blur_variance': blur_var
    }


def validate_leaf_image(metrics, strict_mode=False):
    """
    Validate if image is likely a plant/leaf image.
    
    Args:
        metrics: Dictionary from analyze_image_quality
        strict_mode: If True, apply stricter validation
    
    Returns:
        dict with validation results
    """
    issues = []
    confidence_score = 1.0
    
    # Define thresholds
    if strict_mode:
        min_green_ratio = 0.15
        min_plant_color = 0.20
        max_neon_ratio = 0.05
        max_skin_ratio = 0.10
        min_edge_density = 0.02
        min_blur_var = 100
    else:
        min_green_ratio = 0.08
        min_plant_color = 0.12
        max_neon_ratio = 0.15 
        max_skin_ratio = 0.25
        min_edge_density = 0.01
        min_blur_var = 50
    
    # Check each validation criterion
    if metrics['green_ratio'] < min_green_ratio:
        issues.append(f"Low green content ({metrics['green_ratio']:.2%} < {min_green_ratio:.2%})")
        confidence_score *= 0.7
    
    if metrics['plant_color_ratio'] < min_plant_color:
        issues.append(f"Lacks plant-like colors ({metrics['plant_color_ratio']:.2%} < {min_plant_color:.2%})")
        confidence_score *= 0.6
    
    if metrics['neon_ratio'] > max_neon_ratio:
        issues.append(f"Contains artificial colors ({metrics['neon_ratio']:.2%} > {max_neon_ratio:.2%})")
        confidence_score *= 0.5
    
    if metrics['skin_ratio'] > max_skin_ratio:
        issues.append(f"Contains skin tones - possibly human photo ({metrics['skin_ratio']:.2%} > {max_skin_ratio:.2%})")
        confidence_score *= 0.3
    
    if metrics['blur_variance'] is not None and metrics['blur_variance'] < min_blur_var:
        issues.append(f"Image too blurry ({metrics['blur_variance']:.1f} < {min_blur_var})")
        confidence_score *= 0.8
    
    if metrics['edge_density'] < min_edge_density:
        issues.append(f"Lacks organic edges/texture ({metrics['edge_density']:.3f} < {min_edge_density:.3f})")
        confidence_score *= 0.7
    
    # Brightness checks
    if metrics['brightness'] < 30:
        issues.append("Image too dark")
        confidence_score *= 0.8
    elif metrics['brightness'] > 240:
        issues.append("Image overexposed")
        confidence_score *= 0.8
    
    # Contrast checks
    if metrics['contrast'] < 10:
        issues.append("Image lacks contrast")
        confidence_score *= 0.9
    
    # Color diversity checks
    if metrics['color_diversity'] > 0.8:
        issues.append("Too much color variation - possibly not a natural image")
        confidence_score *= 0.6
    elif metrics['color_diversity'] < 0.01:
        issues.append("Too little color variation - possibly artificial")
        confidence_score *= 0.7
    
    # Overall assessment
    is_likely_leaf = len(issues) <= 2 and confidence_score >= 0.5
    
    # Special cases for definite rejection
    if metrics['skin_ratio'] > 0.4:
        is_likely_leaf = False
        issues.append("High probability of human/animal photo")
    
    if metrics['neon_ratio'] > 0.3:
        is_likely_leaf = False
        issues.append("High probability of artificial/synthetic image")
    
    return {
        'is_likely_leaf': is_likely_leaf,
        'confidence_score': confidence_score,
        'issues': issues,
        'validation_level': 'strict' if strict_mode else 'normal'
    }


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for disease prediction",
            "/health": "GET - Health check",
            "/classes": "GET - Get all supported plant classes"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/classes")
async def get_classes():
    """Get all supported plant disease classes."""
    formatted_classes = []
    for class_name in class_names:
        formatted_info = format_disease_name(class_name)
        formatted_classes.append({
            "raw_class": class_name,
            "formatted_name": formatted_info["formatted_name"],
            "plant": formatted_info["plant"],
            "disease": formatted_info["disease"],
            "is_healthy": formatted_info["is_healthy"]
        })
    
    return {
        "classes": formatted_classes,
        "raw_classes": class_names,
        "total_classes": len(class_names)
    }


@app.post("/predict")
async def predict_disease(
    request: Request,
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.6),
    strict_validation: bool = Form(False),
    skip_validation: bool = Form(False),
    include_advice: bool = Form(True)
):
    """
    Predict plant disease from uploaded image.
    
    Args:
        file: Image file (JPG, JPEG, PNG)
        confidence_threshold: Minimum confidence threshold (0.0-1.0)
        strict_validation: Use stricter validation for leaf detection
        skip_validation: Skip image validation (not recommended)
    
    Returns:
        Prediction results with confidence scores and validation metrics
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess image
        input_arr, pil_img = preprocess_image(image_bytes)
        
        # Comprehensive image analysis
        quality_metrics = analyze_image_quality(pil_img)
        
        # Validate if image is likely a leaf/plant
        validation_result = validate_leaf_image(quality_metrics, strict_validation)
        
        # Get model predictions
        model = load_model()
        predictions = model.predict(input_arr)
        probabilities = predictions[0]
        
        # Get top prediction
        top_idx = int(np.argmax(probabilities))
        top_confidence = float(np.max(probabilities))
        predicted_class = class_names[top_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                "class": class_names[idx],
                "confidence": float(probabilities[idx]),
                "index": int(idx)
            }
            for idx in top_3_indices
        ]
        
        # Determine if we should trust the prediction
        model_confidence_ok = top_confidence >= confidence_threshold
        image_validation_ok = validation_result['is_likely_leaf'] or skip_validation
        
        # Overall prediction validity
        is_valid_prediction = model_confidence_ok and image_validation_ok
        
        # Collect all issues
        all_issues = validation_result['issues'].copy()
        if not model_confidence_ok:
            all_issues.append(f"Model confidence too low ({top_confidence:.2%} < {confidence_threshold:.2%})")
        
        # Format disease info
        formatted_info = format_disease_name(predicted_class)
        
        # Format top 3 predictions with proper names
        formatted_top_3 = []
        for pred in top_3_predictions:
            formatted_pred_info = format_disease_name(pred["class"])
            formatted_top_3.append({
                "class": pred["class"],
                "formatted_name": formatted_pred_info["formatted_name"],
                "plant": formatted_pred_info["plant"],
                "disease": formatted_pred_info["disease"],
                "is_healthy": formatted_pred_info["is_healthy"],
                "confidence": pred["confidence"],
                "index": pred["index"]
            })
        
    # Prepare response
        response = {
            "success": True,
            "prediction": {
                "raw_class": predicted_class,
                "formatted_name": formatted_info["formatted_name"],
                "plant": formatted_info["plant"],
                "disease": formatted_info["disease"],
                "is_healthy": formatted_info["is_healthy"],
                "confidence": round(top_confidence, 4),
                "index": top_idx
            },
            "top_3_predictions": formatted_top_3,
            "validation": {
                "is_valid_prediction": is_valid_prediction,
                "image_validation": {
                    "is_likely_leaf": validation_result['is_likely_leaf'],
                    "confidence_score": round(validation_result['confidence_score'], 3),
                    "validation_level": validation_result['validation_level']
                },
                "model_validation": {
                    "confidence_threshold_met": model_confidence_ok,
                    "confidence_threshold": confidence_threshold
                },
                "issues": all_issues
            },
            "image_metrics": {
                "green_ratio": round(quality_metrics['green_ratio'], 4),
                "plant_color_ratio": round(quality_metrics['plant_color_ratio'], 4),
                "skin_ratio": round(quality_metrics['skin_ratio'], 4),
                "neon_ratio": round(quality_metrics['neon_ratio'], 4),
                "edge_density": round(quality_metrics['edge_density'], 4),
                "brightness": round(quality_metrics['brightness'], 1),
                "contrast": round(quality_metrics['contrast'], 1),
                "color_diversity": round(quality_metrics['color_diversity'], 4),
                "background_score": round(quality_metrics['background_score'], 2),
                "blur_variance": None if quality_metrics['blur_variance'] is None else round(float(quality_metrics['blur_variance']), 2)
            },
            "metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "settings": {
                    "confidence_threshold": confidence_threshold,
                    "strict_validation": strict_validation,
                    "skip_validation": skip_validation
                }
            }
        }

        # Optionally fetch treatment advice when prediction is not healthy and model is confident
        if include_advice and is_valid_prediction and not formatted_info["is_healthy"]:
            advice = await get_treatment_advice(formatted_info["plant"], formatted_info["disease"])
            if advice:
                # advice may contain 'summary' or 'error' and 'source'
                if "summary" in advice:
                    response["treatment_advice"] = {
                        "summary": advice["summary"],
                        "source": advice.get("source", "unknown")
                    }
                    # Add structured data if available
                    if "structured" in advice:
                        response["treatment_advice"]["structured"] = advice["structured"]
                else:
                    response.setdefault("treatment_advice", {})["error"] = advice.get("error", "Unknown error")
            else:
                # Hint to configure Gemini if not available
                if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                    response.setdefault("info", {})["advice_note"] = (
                        "AI advice unavailable: set GEMINI_API_KEY in .env and restart the server."
                    )
        
        # Add warning for invalid images
        if not is_valid_prediction:
            response["warning"] = "This image may not be a plant leaf or the prediction confidence is too low."
            if not image_validation_ok:
                response["suggestion"] = "Please upload a clear image of a plant leaf with good lighting and focus."
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Add a new endpoint for image validation only
@app.post("/validate")
async def validate_image(
    file: UploadFile = File(...),
    strict_mode: bool = Form(False)
):
    """
    Validate if an uploaded image appears to be a plant/leaf image.
    
    Args:
        file: Image file (JPG, JPEG, PNG)
        strict_mode: Use stricter validation criteria
    
    Returns:
        Validation results without disease prediction
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        _, pil_img = preprocess_image(image_bytes)
        
        # Analyze image
        quality_metrics = analyze_image_quality(pil_img)
        validation_result = validate_leaf_image(quality_metrics, strict_mode)
        
        return {
            "is_likely_leaf": validation_result['is_likely_leaf'],
            "confidence_score": round(validation_result['confidence_score'], 3),
            "validation_level": validation_result['validation_level'],
            "issues": validation_result['issues'],
            "metrics": {
                "green_ratio": round(quality_metrics['green_ratio'], 4),
                "plant_color_ratio": round(quality_metrics['plant_color_ratio'], 4),
                "skin_ratio": round(quality_metrics['skin_ratio'], 4),
                "neon_ratio": round(quality_metrics['neon_ratio'], 4),
                "edge_density": round(quality_metrics['edge_density'], 4),
                "brightness": round(quality_metrics['brightness'], 1),
                "contrast": round(quality_metrics['contrast'], 1),
                "color_diversity": round(quality_metrics['color_diversity'], 4),
                "background_score": round(quality_metrics['background_score'], 2),
                "blur_variance": None if quality_metrics['blur_variance'] is None else round(float(quality_metrics['blur_variance']), 2)
            },
            "metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "strict_mode": strict_mode
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)