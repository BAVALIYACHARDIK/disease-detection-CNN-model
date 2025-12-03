import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import cv2  # For blur detection
except Exception:  # noqa: BLE001
    cv2 = None

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model once, with a safe relative path and .keras/.h5 fallback."""
    base_dir = Path(__file__).resolve().parent
    h5_path = base_dir / "trained_model.h5"
    keras_path = base_dir / "trained_model.keras"
    
    # Try .h5 first as it's more compatible with newer TensorFlow
    if h5_path.exists():
        model_path = h5_path
    elif keras_path.exists():
        model_path = keras_path
    else:
        st.error(
            "Model file not found. Expected 'trained_model.keras' or 'trained_model.h5' next to main.py."
        )
        raise FileNotFoundError("Model file not found")

    try:
        # Load with compile=False to avoid optimizer compatibility issues
        import tensorflow.keras.models
        return tensorflow.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {str(e)}")
        st.info("Trying alternative loading method...")
        try:
            # Try legacy loading for older models
            from tensorflow.keras.saving import legacy
            return legacy.load_model(str(model_path), compile=False)
        except Exception as e2:
            st.error(f"Legacy loading also failed: {str(e2)}")
            raise


def preprocess_image(file_obj, target_size=(128, 128)):
    """Load uploaded file into model-ready numpy array and a PIL image for analysis/display."""
    # Reset pointer if it's an UploadedFile used multiple times
    try:
        file_obj.seek(0)
    except Exception:
        pass

    pil_img = Image.open(file_obj).convert("RGB")
    # For model input (keep training preproc consistent: no rescale here)
    model_img = pil_img.resize(target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(model_img)
    input_arr = np.expand_dims(input_arr, axis=0)  # shape (1, H, W, C)
    return input_arr, pil_img


def analyze_image(pil_img):
    """Return simple quality metrics: green pixel ratio and blur variance (if OpenCV available)."""
    np_img = np.asarray(pil_img.resize((224, 224)))
    # Green ratio heuristic: count pixels where G dominates R and B by a margin
    r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
    green_mask = (g > r + 10) & (g > b + 10)
    green_ratio = float(np.count_nonzero(green_mask)) / float(np_img.shape[0] * np_img.shape[1])

    blur_var = None
    if cv2 is not None:
        # Convert to grayscale for Laplacian variance
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return green_ratio, blur_var


def predict_with_confidence(model, input_arr):
    preds = model.predict(input_arr)
    # preds shape: (1, num_classes)
    probs = preds[0]
    top_idx = int(np.argmax(probs))
    top_conf = float(np.max(probs))
    return top_idx, top_conf, probs

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
""")
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    
    st.info("📝 **Tips for best results:**\n"
           "• Upload clear images of plant leaves\n" 
           "• Ensure good lighting and focus\n"
           "• Avoid blurry or very small images\n"
           "• Single leaf images work better than full plants")
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("Validation settings")
        conf_threshold = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)
        min_green_ratio = st.slider("Min green ratio", 0.0, 0.6, 0.10, 0.01)
        blur_threshold = st.number_input("Min blur variance (optional)", value=50.0, step=5.0)

    if test_image is not None:
        with col1:
            st.image(test_image, use_column_width=True)

        # Process and predict
        if st.button("Predict"):
            with st.spinner("Analyzing image and predicting..."):
                try:
                    model = load_model()
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")
                    st.stop()

                try:
                    input_arr, pil_img = preprocess_image(test_image)
                    green_ratio, blur_var = analyze_image(pil_img)
                    top_idx, top_conf, probs = predict_with_confidence(model, input_arr)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.exception(e)
                    st.stop()

                #Define Class
                class_name = ['Apple___Apple_scab',
                    'Apple___Black_rot',
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Blueberry___healthy',
                    'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy',
                    'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot',
                    'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot',
                    'Pepper,_bell___healthy',
                    'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy',
                    'Raspberry___healthy',
                    'Soybean___healthy',
                    'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch',
                    'Strawberry___healthy',
                    'Tomato___Bacterial_spot',
                    'Tomato___Early_blight',
                    'Tomato___Late_blight',
                    'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']

                # Validation logic
                reasons = []
                if top_conf < conf_threshold:
                    reasons.append(f"low confidence ({top_conf:.2f} < {conf_threshold:.2f})")
                if green_ratio < min_green_ratio:
                    reasons.append(f"low green ratio ({green_ratio:.2f} < {min_green_ratio:.2f})")
                if cv2 is not None and blur_var is not None and blur_var < blur_threshold:
                    reasons.append(f"image too blurry (var {blur_var:.1f} < {blur_threshold:.1f})")

                if reasons:
                    st.warning(
                        "Unable to confidently detect a plant disease. This image may not be a leaf or is of low quality (" + ", ".join(reasons) + ")."
                    )
                else:
                    st.success(f"Prediction: {class_name[top_idx]} (confidence {top_conf:.2f})")

                with st.expander("Details"):
                    st.write({
                        "top_index": top_idx,
                        "top_class": class_name[top_idx],
                        "top_confidence": round(top_conf, 4),
                        "green_ratio": round(green_ratio, 4),
                        "blur_variance": None if blur_var is None else round(float(blur_var), 2),
                    })
    else:
        st.info("Upload an image to begin.")
