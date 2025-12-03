import requests
import json
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health Check: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"API not accessible: {e}")
        return False

def get_supported_classes():
    """Get all supported plant classes."""
    try:
        response = requests.get(f"{API_BASE_URL}/classes")
        data = response.json()
        print(f"Supported classes: {data['total_classes']} classes")
        return data['classes']
    except Exception as e:
        print(f"Error getting classes: {e}")
        return None

def predict_plant_disease(image_path, confidence_threshold=0.6, strict_validation=False):
    """
    Send image to API for disease prediction.
    
    Args:
        image_path: Path to image file
        confidence_threshold: Minimum confidence threshold
        strict_validation: Use stricter image validation
    """
    try:
        # Check if file exists
        if not Path(image_path).exists():
            print(f"Image file not found: {image_path}")
            return None
        
        # Prepare the request
        with open(image_path, 'rb') as image_file:
            files = {'file': ('image.jpg', image_file, 'image/jpeg')}
            params = {
                'confidence_threshold': confidence_threshold,
                'strict_validation': strict_validation
            }
            
            # Send request
            response = requests.post(
                f"{API_BASE_URL}/predict",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== PREDICTION RESULTS ===")
            print(f"Success: {result['success']}")
            
            # Prediction results
            prediction = result['prediction']
            print(f"\n🎯 PREDICTION:")
            print(f"Plant Type: {prediction['plant_type']}")
            print(f"Disease Status: {prediction['disease_status']}")
            print(f"Is Healthy: {prediction['is_healthy']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            
            # Top 3 predictions
            print(f"\n📊 TOP 3 PREDICTIONS:")
            for i, pred in enumerate(result['top_3_predictions'], 1):
                print(f"  {i}. {pred['class']} - {pred['confidence']:.2%}")
            
            # Validation results
            validation = result['validation']
            print(f"\n🔍 VALIDATION RESULTS:")
            print(f"Valid Prediction: {validation['is_valid_prediction']}")
            print(f"Image Likely Leaf: {validation['image_validation']['is_likely_leaf']}")
            print(f"Image Confidence: {validation['image_validation']['confidence_score']:.1%}")
            print(f"Model Confidence OK: {validation['model_validation']['confidence_threshold_met']}")
            
            if validation['issues']:
                print(f"\n⚠️  ISSUES DETECTED:")
                for issue in validation['issues']:
                    print(f"  - {issue}")
            
            # Image metrics
            metrics = result['image_metrics']
            print(f"\n📈 IMAGE ANALYSIS:")
            print(f"Green Ratio: {metrics['green_ratio']:.2%}")
            print(f"Plant Colors: {metrics['plant_color_ratio']:.2%}")
            print(f"Skin Tones: {metrics['skin_ratio']:.2%}")
            print(f"Artificial Colors: {metrics['neon_ratio']:.2%}")
            print(f"Edge Density: {metrics['edge_density']:.3f}")
            print(f"Brightness: {metrics['brightness']:.1f}")
            print(f"Contrast: {metrics['contrast']:.1f}")
            if metrics['blur_variance']:
                print(f"Blur Variance: {metrics['blur_variance']}")
            
            # Warning if applicable
            if 'warning' in result:
                print(f"\n⚠️  WARNING: {result['warning']}")
            if 'suggestion' in result:
                print(f"💡 SUGGESTION: {result['suggestion']}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None
            
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def validate_image_only(image_path, strict_mode=False):
    """
    Validate if an image appears to be a plant leaf without making predictions.
    
    Args:
        image_path: Path to image file
        strict_mode: Use stricter validation
    """
    try:
        if not Path(image_path).exists():
            print(f"Image file not found: {image_path}")
            return None
        
        with open(image_path, 'rb') as image_file:
            files = {'file': ('image.jpg', image_file, 'image/jpeg')}
            params = {'strict_mode': strict_mode}
            
            response = requests.post(
                f"{API_BASE_URL}/validate",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== IMAGE VALIDATION RESULTS ===")
            print(f"Is Likely Leaf: {result['is_likely_leaf']}")
            print(f"Confidence Score: {result['confidence_score']:.1%}")
            print(f"Validation Level: {result['validation_level']}")
            
            if result['issues']:
                print(f"\n⚠️  ISSUES:")
                for issue in result['issues']:
                    print(f"  - {issue}")
            else:
                print("\n✅ No issues detected")
            
            metrics = result['metrics']
            print(f"\n📊 DETAILED METRICS:")
            print(f"Green Ratio: {metrics['green_ratio']:.2%}")
            print(f"Plant Colors: {metrics['plant_color_ratio']:.2%}")
            print(f"Skin Detection: {metrics['skin_ratio']:.2%}")
            print(f"Artificial Colors: {metrics['neon_ratio']:.2%}")
            print(f"Edge Density: {metrics['edge_density']:.3f}")
            print(f"Color Diversity: {metrics['color_diversity']:.3f}")
            print(f"Background Score: {metrics['background_score']:.2f}")
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None
            
    except Exception as e:
        print(f"Error validating image: {e}")
        return None

def main():
    """Test the API with comprehensive validation examples."""
    print("=== Plant Disease Detection API Test ===\n")
    
    # Test API health
    if not test_api_health():
        print("API is not running. Start it with: python api.py")
        return
    
    # Get supported classes
    classes = get_supported_classes()
    
    print("\n=== Testing Image Validation ===")
    print("The API now includes comprehensive validation to detect:")
    print("✅ Plant leaves and foliage")
    print("❌ Human photos (skin detection)")
    print("❌ Artificial/synthetic images") 
    print("❌ Non-plant objects")
    print("❌ Blurry or poor quality images")
    print("❌ Images with artificial colors")
    
    print("\n=== Example Usage ===")
    print("1. Basic prediction:")
    print("   result = predict_plant_disease('leaf_image.jpg')")
    
    print("\n2. Strict validation:")
    print("   result = predict_plant_disease('leaf_image.jpg', strict_validation=True)")
    
    print("\n3. Validation only (no prediction):")
    print("   result = validate_image_only('any_image.jpg')")
    
    print("\n=== Test with Your Images ===")
    print("Place test images in this directory and uncomment the lines below:")
    
    # Example usage - uncomment and modify paths to test
    # print("\n--- Testing a plant leaf image ---")
    # result = predict_plant_disease("plant_leaf.jpg")
    
    # print("\n--- Testing a human photo ---")
    # result = validate_image_only("human_photo.jpg")
    
    # print("\n--- Testing with strict validation ---")
    # result = predict_plant_disease("questionable_image.jpg", strict_validation=True)


if __name__ == "__main__":
    main()
