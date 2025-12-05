# Plant Disease Detection & Treatment API

A comprehensive REST API for detecting plant diseases from leaf images using deep learning and providing AI-powered treatment recommendations. This hybrid system combines local machine learning for disease detection with Google's Gemini AI for intelligent treatment advice.

## Key Features

### Disease Detection (Local ML)
- **38 Plant Disease Classes**: Supports detection across Apple, Corn, Grape, Tomato, Potato, Cherry, and 10+ other plant types
- **High Accuracy CNN Model**: Trained TensorFlow model for precise disease classification
- **Advanced Image Validation**: Multi-layered validation including green ratio, plant color detection, skin tone rejection, and blur assessment
- **Confidence Scoring**: Detailed confidence metrics and top-3 predictions
- **Quality Assessment**: Comprehensive image quality analysis with 10+ validation criteria

### AI Treatment Advice (Gemini Integration)
- **Intelligent Treatment Recommendations**: Powered by Google Gemini AI
- **Structured Advice Format**: Organized recommendations with immediate steps, cultural practices, chemical options, and prevention tips
- **Caching System**: Efficient caching with TTL to reduce API calls and improve response times
- **Fallback Mechanisms**: Static advice fallback when Gemini API is unavailable
- **Retry Logic**: Robust error handling with exponential backoff

### API Features
- **RESTful Design**: Clean, intuitive endpoint structure
- **Interactive Documentation**: Auto-generated Swagger UI at `/docs`
- **CORS Support**: Ready for web and mobile integration
- **Comprehensive Validation**: Multiple validation modes (normal/strict)
- **Rich Response Format**: Detailed JSON responses with metadata and metrics

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI Server │───▶│ Local ML Model  │
│ (Web/Mobile/IoT)│    │                  │    │ (TensorFlow)    │
└─────────────────┘    │  ┌─────────────┐ │    └─────────────────┘
                       │  │   Gemini    │ │             │
                       │  │ AI Service  │ │             ▼
                       │  └─────────────┘ │    ┌─────────────────┐
                       └──────────────────┘    │ Disease Result  │
                                │                     │
                                ▼                     │
                       ┌─────────────────┐           │
                       │Treatment Advice │◀──────────┘
                       │   + Caching     │
                       └─────────────────┘
```

## Prerequisites

- **Python**: 3.8 or higher
- **Model File**: `trained_model.keras` or `trained_model.h5`
- **Gemini API Key**: (Optional) For AI treatment advice

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Plant_Disease_Prediction

# Install dependencies
pip install -r requirements_api.txt
```

### 2. Configure Environment (Optional)

Create a `.env` file for Gemini AI integration:

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
ADVICE_CACHE_TTL=3600
```

### 3. Start the Server

```bash
# Method 1: Direct Python
python api.py

# Method 2: Using uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Background process
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
```

### 4. Access the API

- **API Base**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Web Demo**: Open `web_demo.html` in your browser

## API Endpoints

### GET `/`
Returns basic API information, version, and available endpoints.

**Response:**
```json
{
  "message": "Plant Disease Detection API",
  "version": "1.0.0",
  "endpoints": {
    "/predict": "POST - Upload image for disease prediction",
    "/health": "GET - Health check",
    "/classes": "GET - Get all supported plant classes"
  }
}
```

### GET `/health`
Health check endpoint to verify API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET `/classes`
Get all supported plant disease classes with formatted names.

**Response:**
```json
{
  "classes": [
    {
      "raw_class": "Apple___Apple_scab",
      "formatted_name": "Apple - Apple Scab",
      "plant": "Apple",
      "disease": "Apple Scab",
      "is_healthy": false
    }
  ],
  "raw_classes": ["Apple___Apple_scab", "Apple___Black_rot", "..."],
  "total_classes": 38
}
```

### POST `/predict`
**Main endpoint** for disease detection and treatment advice.

**Parameters:**
- `file`: Image file (JPG, JPEG, PNG) - **Required**
- `confidence_threshold`: Minimum confidence (0.0-1.0, default: 0.6)
- `strict_validation`: Use stricter image validation (boolean, default: false)
- `skip_validation`: Skip image validation (boolean, default: false)
- `include_advice`: Include AI treatment advice (boolean, default: true)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@plant_leaf.jpg" \
  -F "confidence_threshold=0.7" \
  -F "include_advice=true"
```

**Example Response with AI Advice:**
```json
{
  "success": true,
  "prediction": {
    "raw_class": "Tomato___Late_blight",
    "formatted_name": "Tomato - Late Blight",
    "plant": "Tomato",
    "disease": "Late Blight",
    "is_healthy": false,
    "confidence": 0.8945,
    "index": 30
  },
  "top_3_predictions": [
    {
      "class": "Tomato___Late_blight",
      "formatted_name": "Tomato - Late Blight",
      "plant": "Tomato",
      "disease": "Late Blight",
      "is_healthy": false,
      "confidence": 0.8945,
      "index": 30
    }
  ],
  "treatment_advice": {
    "summary": "Late blight treatment requires immediate action...",
    "source": "gemini",
    "structured": {
      "short_summary": "Late blight is a serious fungal disease requiring immediate removal of infected tissue and fungicide application.",
      "immediate_steps": [
        "Remove and destroy all infected plant parts immediately",
        "Improve air circulation around plants",
        "Stop overhead watering"
      ],
      "cultural_practices": [
        "Water at soil level, avoid wetting leaves",
        "Space plants for good air flow",
        "Remove plant debris regularly",
        "Use resistant varieties when replanting"
      ],
      "chemical_options": [
        {
          "active_ingredient": "Copper compounds",
          "notes": "Effective preventive treatment, apply before infection"
        },
        {
          "active_ingredient": "Chlorothalonil",
          "notes": "Protective fungicide for early treatment"
        }
      ],
      "prevention": [
        "Plant resistant tomato varieties",
        "Ensure proper plant spacing",
        "Monitor weather conditions",
        "Apply preventive treatments during humid periods"
      ]
    }
  },
  "validation": {
    "is_valid_prediction": true,
    "image_validation": {
      "is_likely_leaf": true,
      "confidence_score": 0.892,
      "validation_level": "normal"
    },
    "model_validation": {
      "confidence_threshold_met": true,
      "confidence_threshold": 0.7
    },
    "issues": []
  },
  "image_metrics": {
    "green_ratio": 0.3456,
    "plant_color_ratio": 0.4123,
    "skin_ratio": 0.0045,
    "neon_ratio": 0.0012,
    "edge_density": 0.0892,
    "brightness": 142.5,
    "contrast": 45.8,
    "color_diversity": 0.2341,
    "background_score": 0.75,
    "blur_variance": 189.34
  },
  "metadata": {
    "filename": "plant_leaf.jpg",
    "content_type": "image/jpeg",
    "settings": {
      "confidence_threshold": 0.7,
      "strict_validation": false,
      "skip_validation": false
    }
  }
}
```

### POST `/validate`
Validate if an image appears to be a plant leaf without running disease prediction.

**Parameters:**
- `file`: Image file (JPG, JPEG, PNG) - **Required**
- `strict_mode`: Use stricter validation criteria (boolean, default: false)

**Example Response:**
```json
{
  "is_likely_leaf": true,
  "confidence_score": 0.856,
  "validation_level": "normal",
  "issues": [],
  "metrics": {
    "green_ratio": 0.3456,
    "plant_color_ratio": 0.4123,
    "skin_ratio": 0.0045
  }
}
```
  "prediction": {
    "class": "Tomato___Late_blight",
    "plant_type": "Tomato",
    "disease_status": "Late_blight",
    "is_healthy": false,
    "confidence": 0.8945,
    "index": 30
  },
  "top_3_predictions": [
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.8945,
      "index": 30
    },
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.0678,
      "index": 29
    },
    {
      "class": "Tomato___Leaf_Mold",
      "confidence": 0.0234,
      "index": 31
    }
  ],
  "quality_metrics": {
    "is_valid": true,
    "green_ratio": 0.3456,
    "blur_variance": 89.12,
    "issues": []
  },
  "metadata": {
    "filename": "plant_image.jpg",
    "content_type": "image/jpeg",
    "thresholds": {
      "confidence": 0.7,
      "green_ratio": 0.10,
      "blur": 50.0
    }
  }
}
```

## Testing the API

### 1. Using the Enhanced Web Demo
Open `web_demo.html` in your browser for a beautiful, interactive interface that displays:
- Disease detection results
- Structured treatment advice with organized sections
- Copy-to-clipboard functionality
- Image validation feedback
- Confidence metrics visualization

### 2. Using the Test Scripts
```bash
# Test basic API functionality
python test_api.py

# Test Gemini integration (requires API key)
python test_gemini.py
```

### 3. Using curl Commands

```bash
# Health check
curl http://localhost:8000/health

# Get all supported classes
curl http://localhost:8000/classes

# Basic disease prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_leaf.jpg"

# Prediction with AI treatment advice
curl -X POST "http://localhost:8000/predict" \
  -F "file=@diseased_leaf.jpg" \
  -F "confidence_threshold=0.8" \
  -F "include_advice=true"

# Image validation only
curl -X POST "http://localhost:8000/validate" \
  -F "file=@test_image.jpg" \
  -F "strict_mode=true"
```

### 4. Using Python requests

```python
import requests

def predict_with_advice(image_path):
    """Complete prediction with AI treatment advice"""
    with open(image_path, 'rb') as f:
        files = {'file': ('image.jpg', f, 'image/jpeg')}
        data = {
            'confidence_threshold': 0.7,
            'include_advice': True,
            'strict_validation': False
        }
        response = requests.post(
            'http://localhost:8000/predict',
            files=files,
            data=data
        )
        return response.json()

# Example usage
result = predict_with_advice('plant_leaf.jpg')
if result['success']:
    prediction = result['prediction']
    print(f"Disease: {prediction['formatted_name']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    # Check for AI advice
    if 'treatment_advice' in result:
        advice = result['treatment_advice']
        if 'structured' in advice:
            structured = advice['structured']
            print(f"Summary: {structured['short_summary']}")
            print("Immediate steps:")
            for step in structured['immediate_steps']:
                print(f"  - {step}")
```

## Integration Examples

### Web Application (JavaScript)
```javascript
async function detectDiseaseWithAdvice(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('include_advice', 'true');
    formData.append('confidence_threshold', '0.7');
    
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Display disease information
            console.log(`Disease: ${result.prediction.formatted_name}`);
            console.log(`Confidence: ${(result.prediction.confidence * 100).toFixed(1)}%`);
            
            // Display treatment advice if available
            if (result.treatment_advice && result.treatment_advice.structured) {
                const advice = result.treatment_advice.structured;
                displayTreatmentAdvice(advice);
            }
        }
        
        return result;
    } catch (error) {
        console.error('Prediction failed:', error);
        throw error;
    }
}

function displayTreatmentAdvice(advice) {
    console.log('Treatment Summary:', advice.short_summary);
    console.log('Immediate Steps:', advice.immediate_steps);
    console.log('Prevention Tips:', advice.prevention);
}
```

### React Component
```jsx
import React, { useState } from 'react';

const PlantDiseaseDetector = () => {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleImageUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('include_advice', 'true');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            
            {loading && <p>Analyzing image...</p>}
            
            {result?.success && (
                <div>
                    <h3>Detection Result</h3>
                    <p><strong>Disease:</strong> {result.prediction.formatted_name}</p>
                    <p><strong>Confidence:</strong> {(result.prediction.confidence * 100).toFixed(1)}%</p>
                    
                    {result.treatment_advice?.structured && (
                        <div>
                            <h4>Treatment Advice</h4>
                            <p>{result.treatment_advice.structured.short_summary}</p>
                            
                            <h5>Immediate Steps:</h5>
                            <ul>
                                {result.treatment_advice.structured.immediate_steps.map((step, i) => (
                                    <li key={i}>{step}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default PlantDiseaseDetector;
```

### Mobile App (React Native)
```javascript
import { launchImageLibrary } from 'react-native-image-picker';

const PlantDiseaseService = {
    async detectDisease(imageUri) {
        const formData = new FormData();
        formData.append('file', {
            uri: imageUri,
            type: 'image/jpeg',
            name: 'plant_leaf.jpg'
        });
        formData.append('include_advice', 'true');
        formData.append('confidence_threshold', '0.7');

        try {
            const response = await fetch('http://your-api-url:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                body: formData
            });
            
            const result = await response.json();
            return result;
        } catch (error) {
            throw new Error(`Detection failed: ${error.message}`);
        }
    }
};

// Usage in component
const selectAndAnalyzeImage = async () => {
    launchImageLibrary({ mediaType: 'photo' }, async (response) => {
        if (response.assets && response.assets[0]) {
            try {
                const result = await PlantDiseaseService.detectDisease(
                    response.assets[0].uri
                );
                // Handle result with disease detection and treatment advice
                console.log('Disease:', result.prediction.formatted_name);
                if (result.treatment_advice?.structured) {
                    console.log('Advice:', result.treatment_advice.structured);
                }
            } catch (error) {
                console.error('Analysis failed:', error);
            }
        }
    });
};
```

### Python Application
```python
import requests
import json
from pathlib import Path

class PlantDiseaseAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def detect_disease(self, image_path, include_advice=True, confidence_threshold=0.7):
        """Detect plant disease with optional AI treatment advice"""
        with open(image_path, 'rb') as image_file:
            files = {'file': ('image.jpg', image_file, 'image/jpeg')}
            data = {
                'confidence_threshold': confidence_threshold,
                'include_advice': include_advice,
                'strict_validation': False
            }
            
            response = requests.post(
                f'{self.base_url}/predict',
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def validate_image(self, image_path, strict_mode=False):
        """Validate if image is suitable for plant disease detection"""
        with open(image_path, 'rb') as image_file:
            files = {'file': ('image.jpg', image_file, 'image/jpeg')}
            data = {'strict_mode': strict_mode}
            
            response = requests.post(
                f'{self.base_url}/validate',
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()

# Example usage
api = PlantDiseaseAPI()

# Detect disease with treatment advice
result = api.detect_disease('plant_leaf.jpg')
if result['success']:
    prediction = result['prediction']
    print(f"Plant: {prediction['plant']}")
    print(f"Disease: {prediction['disease']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    
    # Display treatment advice
    if 'treatment_advice' in result and 'structured' in result['treatment_advice']:
        advice = result['treatment_advice']['structured']
        print(f"\nTreatment Summary: {advice['short_summary']}")
        print("\nImmediate Actions:")
        for step in advice['immediate_steps']:
            print(f"  • {step}")
```

## Configuration & Setup

### Environment Variables

Create a `.env` file in the project root for advanced configuration:

```bash
# Required for AI treatment advice
GEMINI_API_KEY=your_google_gemini_api_key_here

# Optional: Gemini model selection
GEMINI_MODEL=gemini-1.5-flash  # or gemini-1.5-pro

# Optional: Cache settings
ADVICE_CACHE_TTL=3600  # seconds (1 hour)

# Optional: API settings
API_HOST=0.0.0.0
API_PORT=8000
```

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file as `GEMINI_API_KEY`

### Model Files

The API automatically detects and loads model files in this priority order:
1. `trained_model.keras` (preferred)
2. `trained_model.h5` (fallback)

Ensure one of these files exists in the project root directory.

### Dependencies

The project uses different requirement files for specific features:

```bash
# Core API functionality (disease detection)
pip install -r requirements_api.txt

# Gemini AI integration (treatment advice)
pip install -r requirements_gemini.txt

# Complete installation (recommended)
pip install -r requirements_api.txt -r requirements_gemini.txt
```

**Key Dependencies:**
- `fastapi`: Web API framework
- `tensorflow==2.10.0`: Machine learning model
- `protobuf==3.19.6`: TensorFlow compatibility
- `httpx>=0.27.0`: Gemini API client
- `pillow`: Image processing
- `opencv-python`: Advanced image validation
- `uvicorn`: ASGI server

## Deployment Options

### 1. Local Development
```bash
# Development server with auto-reload
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Production server
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_api.txt requirements_gemini.txt ./
RUN pip install --no-cache-dir -r requirements_api.txt -r requirements_gemini.txt

# Copy application code and model
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run Docker container
docker build -t plant-disease-api .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here plant-disease-api
```

### 3. Cloud Deployment

#### Heroku
```yaml
# Procfile
web: uvicorn api:app --host 0.0.0.0 --port $PORT

# runtime.txt
python-3.9.19
```

#### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/plant-disease-api', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/plant-disease-api']
```

#### AWS Lambda (with Mangum)
```python
# lambda_handler.py
from mangum import Mangum
from api import app

handler = Mangum(app)
```

### 4. Production Considerations

```bash
# Use a production ASGI server
pip install gunicorn

# Run with Gunicorn + Uvicorn workers
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With environment file
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --env-file .env
```

**Production Settings:**
- Use environment variables for secrets
- Enable HTTPS/SSL termination
- Configure proper CORS origins
- Set up monitoring and logging
- Use a reverse proxy (nginx/Apache)
- Configure rate limiting

## Supported Plant Classes & Diseases

The API supports detection of **38 different plant disease classes** across **14 plant types**:

### Apple (4 classes)
- **Apple Scab** - Fungal disease causing dark spots on leaves
- **Black Rot** - Causes brown circular spots and fruit rot
- **Cedar Apple Rust** - Orange spots on leaves, requires cedar trees nearby
- **Healthy** - No disease detected

### Blueberry (1 class)
- **Healthy** - No disease detected

### Cherry (2 classes)
- **Powdery Mildew** - White powdery coating on leaves
- **Healthy** - No disease detected

### Corn/Maize (4 classes)
- **Cercospora Leaf Spot (Gray Leaf Spot)** - Gray rectangular spots
- **Common Rust** - Orange/brown pustules on leaves
- **Northern Leaf Blight** - Large cigar-shaped lesions
- **Healthy** - No disease detected

### Grape (4 classes)
- **Black Rot** - Dark circular spots on leaves and fruit
- **Esca (Black Measles)** - Tiger stripe pattern on leaves
- **Leaf Blight (Isariopsis Leaf Spot)** - Brown spots with halos
- **Healthy** - No disease detected

### Orange (1 class)
- **Huanglongbing (Citrus Greening)** - Yellowing, stunted growth

### Peach (2 classes)
- **Bacterial Spot** - Small dark spots on leaves
- **Healthy** - No disease detected

### Pepper (Bell) (2 classes)
- **Bacterial Spot** - Small dark lesions on leaves
- **Healthy** - No disease detected

### Potato (3 classes)
- **Early Blight** - Dark concentric ring spots
- **Late Blight** - Water-soaked spots, white mold
- **Healthy** - No disease detected

### Strawberry (2 classes)
- **Leaf Scorch** - Brown spots with purple margins
- **Healthy** - No disease detected

### Tomato (10 classes)
- **Bacterial Spot** - Small dark spots with yellow halos
- **Early Blight** - Dark spots with concentric rings
- **Late Blight** - Large brown spots with white mold
- **Leaf Mold** - Yellow spots turning brown
- **Septoria Leaf Spot** - Small circular spots with dark borders
- **Spider Mites (Two-spotted)** - Stippling and webbing
- **Target Spot** - Brown spots with light centers
- **Tomato Yellow Leaf Curl Virus** - Yellowing and curling leaves
- **Tomato Mosaic Virus** - Mottled yellow-green pattern
- **Healthy** - No disease detected

### Other Plants (3 classes)
- **Raspberry - Healthy**
- **Soybean - Healthy**
- **Squash - Powdery Mildew**

## Image Validation System

The API includes comprehensive image validation to ensure accurate predictions:

### Validation Criteria
- **Green Ratio**: Percentage of green pixels (plant-like colors)
- **Plant Color Detection**: Natural plant colors (green variations, brown, yellow)
- **Skin Tone Rejection**: Prevents human/animal photos from being processed
- **Neon Color Detection**: Rejects artificial/synthetic images
- **Edge Analysis**: Detects organic leaf shapes and textures
- **Blur Detection**: Ensures image sharpness for accurate prediction
- **Background Analysis**: Identifies solid backgrounds typical of plant photos
- **Brightness/Contrast**: Validates proper lighting conditions

### Validation Modes
- **Normal Mode**: Standard validation suitable for most use cases
- **Strict Mode**: Enhanced validation for critical applications
- **Skip Validation**: Bypass validation (not recommended for production)

## AI Treatment Advice System

### How It Works
1. **Disease Detection**: Local ML model identifies the plant disease
2. **AI Query**: If disease detected, system queries Gemini AI for treatment advice
3. **Structured Response**: AI returns organized treatment recommendations
4. **Caching**: Advice is cached to improve performance and reduce API costs
5. **Fallback**: Static advice available when AI is unavailable

### Advice Structure
```json
{
  "short_summary": "Brief disease overview and urgency level",
  "immediate_steps": ["Urgent actions to take now"],
  "cultural_practices": ["Organic and cultural control methods"],
  "chemical_options": [
    {
      "active_ingredient": "Compound name",
      "notes": "Usage notes and effectiveness"
    }
  ],
  "prevention": ["Future prevention strategies"]
}
```

### Reliability Features
- **Retry Logic**: 3 attempts with exponential backoff
- **Rate Limit Handling**: Respects API rate limits
- **Error Recovery**: Graceful fallback to static advice
- **Caching**: 1-hour TTL to reduce redundant API calls

## Troubleshooting & FAQ

### Common Issues & Solutions

#### Model Loading Issues
**Problem**: `Model file not found` error
```bash
FileNotFoundError: Model file not found. Expected 'trained_model.keras' or 'trained_model.h5'
```
**Solution**: 
- Ensure model file exists in project root
- Check file permissions
- Verify file isn't corrupted

#### API Connection Issues
**Problem**: API not accessible or CORS errors
**Solutions**:
- Check server is running: `curl http://localhost:8000/health`
- Verify port availability: `netstat -an | grep 8000`
- For CORS issues, update `allow_origins` in CORS middleware

#### Gemini API Issues
**Problem**: Treatment advice not working
**Solutions**:
- Verify `GEMINI_API_KEY` is set in `.env` file
- Check API key validity at [Google AI Studio](https://makersuite.google.com/app/apikey)
- Ensure sufficient API quota
- Check network connectivity

#### Image Validation Failures
**Problem**: "Image may not be a plant leaf" warnings
**Solutions**:
- Use clear, well-lit photos of plant leaves
- Ensure leaves occupy majority of image
- Avoid photos with dominant background colors
- Use `skip_validation=true` for testing (not recommended for production)

#### Low Confidence Predictions
**Problem**: Predictions below confidence threshold
**Solutions**:
- Improve image quality (lighting, focus, resolution)
- Ensure leaf is clearly visible and disease symptoms are apparent
- Lower confidence threshold for testing
- Try different angles or lighting conditions

#### Performance Issues
**Problem**: Slow response times or memory errors
**Solutions**:
- Reduce image size before upload (< 2MB recommended)
- Use JPEG format instead of PNG
- Consider using GPU for inference in production
- Monitor memory usage and restart API if needed

### Frequently Asked Questions

#### Q: Can I use this API without the Gemini integration?
**A**: Yes! The API works perfectly for disease detection without Gemini. Simply don't set the `GEMINI_API_KEY` or set `include_advice=false` in requests.

#### Q: How accurate is the disease detection?
**A**: The model achieves high accuracy on the trained dataset. However, real-world performance depends on image quality, lighting, and similarity to training data.

#### Q: Can I add new plant diseases?
**A**: You would need to retrain the model with new disease classes. The current model is trained on the specific 38 classes listed.

#### Q: Is the API suitable for commercial use?
**A**: Yes, but consider:
- Implementing rate limiting
- Using proper authentication
- Monitoring API usage and costs (especially Gemini calls)
- Adding comprehensive logging

#### Q: How do I handle multiple images at once?
**A**: The API processes one image per request. For batch processing, send multiple parallel requests or implement a queue system.

#### Q: Can I run this offline?
**A**: Disease detection works offline. Treatment advice requires internet for Gemini API calls, but falls back to static advice when unavailable.

### Performance Optimization Tips

1. **Image Preprocessing**:
   - Resize images to reasonable dimensions before upload
   - Use JPEG compression to reduce file sizes
   - Ensure good lighting and focus

2. **API Configuration**:
   - Adjust confidence thresholds based on your use case
   - Use caching for frequently accessed advice
   - Consider implementing client-side caching

3. **Production Deployment**:
   - Use multiple worker processes
   - Implement connection pooling
   - Set up proper monitoring and logging
   - Use a CDN for static assets

4. **Cost Management**:
   - Monitor Gemini API usage
   - Implement rate limiting for advice requests
   - Use caching to reduce redundant AI calls

### Error Codes & Meanings

| Status Code | Meaning | Solution |
|-------------|---------|----------|
| 400 | Bad Request - Invalid file type | Upload JPG, JPEG, or PNG files |
| 422 | Validation Error - Invalid parameters | Check parameter types and ranges |
| 500 | Internal Server Error - Model/processing error | Check logs, restart API if needed |
| 413 | Payload Too Large - File too big | Reduce image file size |

### Monitoring & Logging

Enable detailed logging for troubleshooting:

```python
import logging

# Add to api.py for detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions and errors
logger.info(f"Prediction made: {prediction_result}")
logger.error(f"Gemini API error: {error_message}")
```

### Getting Help

- **Documentation**: Check this README and inline API docs at `/docs`
- **Testing**: Use the included test scripts and web demo
- **Issues**: Report bugs with detailed error messages and steps to reproduce
- **Feature Requests**: Suggest improvements with clear use cases

## License & Legal

This project is provided for educational and research purposes. When using in production:

- Ensure compliance with Google Gemini API terms of service
- Consider privacy implications of image uploads
- Implement appropriate data retention policies
- Add proper attribution for the AI components

## Contributing

Contributions are welcome! Areas for improvement:

- Additional plant disease classes
- Enhanced image validation algorithms  
- Performance optimizations
- Multi-language support for treatment advice
- Mobile app examples
- Docker improvements

## Support & Community

- **Issues**: Create detailed bug reports with logs and reproduction steps
- **Discussions**: Share use cases and integration examples
- **Documentation**: Help improve this README and API docs
- **Testing**: Contribute test cases and validation datasets

---

**Built for the agricultural and plant health community**
