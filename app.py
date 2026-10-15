import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

st.set_page_config(
    page_title="CFRP Thermal Defect Segmentation",
    page_icon="🔍",
    layout="wide"
)

MODEL_PATH = "unet_best.keras"

@st.cache_resource
def load_segmentation_model(path: str):
    """Loads the trained Keras U-Net model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found in the workspace.")
    # compile=False allows loading even if custom metrics/losses were defined in Colab
    return tf.keras.models.load_model(path, compile=False)

def preprocess_thermal_image(image: Image.Image, norm_mode: str = "Standard [0, 1]"):
    """
    Preprocesses uploaded thermal image to (1, 256, 256, 1).
    Preserves thermal contrast and handles grayscale conversion.
    """
    # Convert to single-channel grayscale
    img_gray = np.array(image.convert("L"), dtype=np.float32)
    
    # Resize to model input shape (256, 256)
    img_resized = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Normalization selection (protects against assuming without checking)
    if norm_mode == "Standard [0, 1]":
        img_norm = img_resized / 255.0
    elif norm_mode == "Per-Image Min-Max":
        min_v, max_v = img_resized.min(), img_resized.max()
        img_norm = (img_resized - min_v) / (max_v - min_v + 1e-7)
    elif norm_mode == "Z-Score (Standardized)":
        mean_v, std_v = img_resized.mean(), img_resized.std()
        img_norm = (img_resized - mean_v) / (std_v + 1e-7)
    else:
        img_norm = img_resized / 255.0

    # Expand dims to (1, 256, 256, 1)
    tensor_in = np.expand_dims(img_norm, axis=(0, -1))
    return tensor_in, img_resized.astype(np.uint8)

def create_defect_overlay(original_gray_256: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """Overlays the detected defect regions on the original thermal image."""
    base_bgr = cv2.cvtColor(original_gray_256, cv2.COLOR_GRAY2BGR)
    color_mask = np.zeros_like(base_bgr)
    # Highlight defects in bright red / thermal warning color (BGR: 0, 0, 255)
    color_mask[mask > 0] = [0, 0, 255]
    
    overlay = cv2.addWeighted(base_bgr, 1.0, color_mask, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ----------------- UI Layout -----------------
st.title("🔬 CFRP Thermal Defect Segmentation")
st.markdown("Automated defect detection in Carbon Fiber Reinforced Polymer (CFRP) using Infrared Thermography and Deep Learning.")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Model & Inference Settings")
    norm_option = st.selectbox(
        "Normalization Strategy",
        ["Standard [0, 1]", "Per-Image Min-Max", "Z-Score (Standardized)"],
        index=0,
        help="Defaults to [0, 1] scaling. We will lock this to your notebook's exact preprocessing once inspected."
    )
    conf_threshold = st.slider(
        "Defect Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Probability cutoff for binary defect classification."
    )
    overlay_opacity = st.slider("Overlay Opacity", 0.1, 1.0, 0.5, 0.1)

# Load model
try:
    with st.spinner("Loading U-Net model (`unet_best.keras`)..."):
        model = load_segmentation_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.text(f"Expected Input: {model.input_shape}")
except Exception as e:
    st.error(f"Error loading model `{MODEL_PATH}`: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a CFRP Thermal Image (Grayscale PNG, JPG, TIFF, BMP)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

if uploaded_file is not None:
    try:
        raw_image = Image.open(uploaded_file)
        
        # Preprocessing
        input_tensor, original_gray_256 = preprocess_thermal_image(raw_image, norm_mode=norm_option)
        
        # Inference
        with st.spinner("Running U-Net segmentation..."):
            pred_probs = model.predict(input_tensor, verbose=0)  # Shape: (1, 256, 256, 1)
        
        # Squeeze to 2D
        pred_map = np.squeeze(pred_probs[0])
        binary_mask = (pred_map >= conf_threshold).astype(np.uint8) * 255
        
        # Metrics
        defect_pixel_count = int(np.sum(binary_mask > 0))
        defect_percentage = (defect_pixel_count / (256 * 256)) * 100
        
        # Overlay
        overlay_rgb = create_defect_overlay(original_gray_256, binary_mask, alpha=overlay_opacity)
        
        # Display side-by-side
        st.subheader("📊 Segmentation Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(original_gray_256, caption="Preprocessed Input (256x256 Grayscale)", use_container_width=True, clamp=True)
        with col2:
            st.image(binary_mask, caption=f"Predicted Defect Mask (Thresh: {conf_threshold})", use_container_width=True, clamp=True)
        with col3:
            st.image(overlay_rgb, caption="Defect Overlay (Red)", use_container_width=True)

        st.metric("Detected Defect Area", f"{defect_percentage:.2f}% of total surface", f"{defect_pixel_count} px")
        
        if defect_pixel_count > 0:
            st.warning("⚠️ Subsurface defect / delamination detected.")
        else:
            st.success("✅ No defects detected above the confidence threshold.")

    except Exception as e:
        st.error(f"Inference pipeline error: {e}")
else:
    st.info("👆 Please upload a thermal inspection image to begin.")