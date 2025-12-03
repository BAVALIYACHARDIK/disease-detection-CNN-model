# 🛡️ Plant Disease API - Image Validation Guide

## Overview

The Plant Disease Detection API now includes comprehensive image validation to ensure uploaded images are actually plant leaves. This prevents incorrect predictions on non-plant images.

## What the Validation Detects

### ✅ **Valid Plant Images**
- Plant leaves (healthy or diseased)
- Various leaf colors (green, brown, yellow)
- Natural organic shapes and textures
- Appropriate lighting and focus

### ❌ **Invalid Images Detected**
1. **Human/Animal Photos**
   - Skin tone detection
   - Facial features
   - Body parts

2. **Artificial/Synthetic Images**
   - Computer graphics
   - Cartoons/drawings
   - Images with neon/artificial colors

3. **Non-Plant Objects**
   - Buildings, vehicles, electronics
   - Food items (non-plant)
   - Abstract patterns

4. **Poor Quality Images**
   - Blurry or out-of-focus
   - Too dark or overexposed
   - Lack of detail/texture

## API Endpoints

### 1. `/predict` - Disease Prediction with Validation

**Parameters:**
- `file`: Image file (required)
- `confidence_threshold`: Minimum prediction confidence (default: 0.6)
- `strict_validation`: Use stricter validation (default: false)
- `skip_validation`: Skip image validation (default: false, not recommended)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@plant_image.jpg" \
  -F "confidence_threshold=0.7" \
  -F "strict_validation=true"
```

**Response Structure:**
```json
{
  "success": true,
  "prediction": {
    "class": "Tomato___Late_blight",
    "plant_type": "Tomato",
    "disease_status": "Late_blight",
    "is_healthy": false,
    "confidence": 0.8945
  },
  "validation": {
    "is_valid_prediction": true,
    "image_validation": {
      "is_likely_leaf": true,
      "confidence_score": 0.85,
      "validation_level": "strict"
    },
    "model_validation": {
      "confidence_threshold_met": true
    },
    "issues": []
  },
  "image_metrics": {
    "green_ratio": 0.3456,
    "plant_color_ratio": 0.4123,
    "skin_ratio": 0.0012,
    "neon_ratio": 0.0001,
    "edge_density": 0.045,
    "brightness": 145.3,
    "contrast": 52.7,
    "blur_variance": 89.12
  }
}
```

### 2. `/validate` - Image Validation Only

**Parameters:**
- `file`: Image file (required)
- `strict_mode`: Use stricter validation (default: false)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/validate" \
  -F "file=@test_image.jpg" \
  -F "strict_mode=true"
```

## Validation Metrics Explained

### **Green Ratio**
- Percentage of pixels that are predominantly green
- **Normal threshold:** ≥ 8%
- **Strict threshold:** ≥ 15%

### **Plant Color Ratio**
- Percentage of pixels with plant-like colors (green, brown, yellow)
- **Normal threshold:** ≥ 12%
- **Strict threshold:** ≥ 20%

### **Skin Ratio**
- Percentage of pixels matching human skin tones
- **Normal threshold:** ≤ 25%
- **Strict threshold:** ≤ 10%

### **Neon/Artificial Color Ratio**
- Percentage of pixels with artificial/synthetic colors
- **Normal threshold:** ≤ 15%
- **Strict threshold:** ≤ 5%

### **Edge Density**
- Measure of organic shapes and textures
- **Normal threshold:** ≥ 0.01
- **Strict threshold:** ≥ 0.02

### **Blur Variance**
- Image sharpness measurement
- **Normal threshold:** ≥ 50
- **Strict threshold:** ≥ 100

## Usage Examples

### Python Client
```python
import requests

# Basic prediction with validation
def predict_with_validation(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/predict',
            files={'file': f},
            params={'confidence_threshold': 0.7}
        )
    return response.json()

# Strict validation
def strict_prediction(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/predict',
            files={'file': f},
            params={
                'confidence_threshold': 0.8,
                'strict_validation': True
            }
        )
    return response.json()

# Validation only
def validate_image(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/validate',
            files={'file': f},
            params={'strict_mode': True}
        )
    return response.json()
```

### JavaScript/Web Client
```javascript
async function predictDisease(imageFile, strict = false) {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('confidence_threshold', '0.7');
    formData.append('strict_validation', strict);
    
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

async function validateImage(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('strict_mode', 'true');
    
    const response = await fetch('http://localhost:8000/validate', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}
```

## Handling Different Image Types

### **Valid Plant Images**
```json
{
  "is_valid_prediction": true,
  "image_validation": {
    "is_likely_leaf": true,
    "confidence_score": 0.85
  },
  "issues": []
}
```

### **Human Photo Detected**
```json
{
  "is_valid_prediction": false,
  "image_validation": {
    "is_likely_leaf": false,
    "confidence_score": 0.15
  },
  "issues": [
    "Contains skin tones - possibly human photo (45.2% > 25.0%)",
    "Low green content (2.1% < 8.0%)"
  ],
  "warning": "This image may not be a plant leaf or the prediction confidence is too low.",
  "suggestion": "Please upload a clear image of a plant leaf with good lighting and focus."
}
```

### **Artificial Image Detected**
```json
{
  "is_valid_prediction": false,
  "image_validation": {
    "is_likely_leaf": false,
    "confidence_score": 0.25
  },
  "issues": [
    "Contains artificial colors (32.1% > 15.0%)",
    "Too much color variation - possibly not a natural image",
    "Lacks plant-like colors (5.2% < 12.0%)"
  ]
}
```

## Best Practices

### **For Web/Mobile Applications**
1. **Pre-validate images** using the `/validate` endpoint before prediction
2. **Show clear error messages** when invalid images are detected
3. **Provide guidance** on what constitutes a good plant image
4. **Use normal validation** for general use, strict for critical applications

### **Error Handling**
```python
def safe_predict(image_path):
    try:
        # First validate the image
        validation = validate_image(image_path)
        
        if not validation['is_likely_leaf']:
            print("❌ Not a plant image:")
            for issue in validation['issues']:
                print(f"  - {issue}")
            return None
        
        # If validation passes, make prediction
        result = predict_with_validation(image_path)
        
        if not result['validation']['is_valid_prediction']:
            print("⚠️ Low confidence prediction")
            return result
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### **User Interface Guidelines**
1. **Clear upload instructions:** "Upload a clear photo of a plant leaf"
2. **Visual examples:** Show good vs. bad image examples
3. **Progressive validation:** Validate on upload, before sending to API
4. **Helpful error messages:** Explain why an image was rejected
5. **Retry mechanism:** Allow users to upload different images

## Configuration Recommendations

### **Strict Mode (Medical/Research Applications)**
- Use for critical applications requiring high accuracy
- Lower false positive rate
- May reject some valid but challenging images

### **Normal Mode (General Use)**
- Balanced approach for most applications
- Good balance of accuracy and usability
- Accepts wider range of plant images

### **Skip Validation (Development Only)**
- Only for testing and development
- Not recommended for production
- Allows any image type through

## Troubleshooting

### **Common Issues**

1. **"Low green content" for diseased plants**
   - Some diseases cause browning/yellowing
   - Use normal mode instead of strict
   - The plant_color_ratio metric includes brown/yellow

2. **Good plant images rejected**
   - Check image quality (blur, lighting)
   - Ensure image shows actual leaves, not just stems
   - Try normal validation mode

3. **Background interference**
   - Images with busy backgrounds may be rejected
   - Use images with simple backgrounds when possible
   - The background_score metric helps detect this

4. **False positives on artificial plants**
   - Very realistic artificial plants may pass validation
   - Consider the edge_density and texture metrics
   - Use strict mode for better detection

Remember: The validation system is designed to catch obvious non-plant images. For edge cases, manual review may still be necessary.
