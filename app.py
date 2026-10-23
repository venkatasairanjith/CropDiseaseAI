"""
AI-Based Crop Disease Detection and Management System
Alliance University - E1CSA313 Design Project - 1
Team: Sai Ranjith, A. Akhileswara achari, S. Haneesh Ranganath, D. Karunakar
Mentor: Dr. Devika P
Interactive Web Dashboard implementing Proposed Architecture:
- Part 1: Crop Leaf Image Acquisition
- Part 2: Image Preprocessing (OpenCV)
- Part 3: AI Disease Classification & Severity Estimation
- Part 4: Multimodal Crop Health Assessment
"""

import streamlit as st
import numpy as np
from PIL import Image

from core.part1_acquisition import acquire_leaf_image, BENCHMARK_SAMPLES
from core.part2_preprocessing import preprocess_leaf_image
from core.part3_classifier import CropDiseaseClassifier, classify_disease_and_severity
from core.part4_assessment import assess_crop_health

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Crop Disease & Health Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f1f8e9;
        border-left: 5px solid #4caf50;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .alert-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffb300;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .danger-box {
        background-color: #ffebee;
        border-left: 5px solid #e53935;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌾 AI-Based Crop Disease Detection & Management System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header"><b>Alliance University</b> | E1CSA313 Design Project - 1 | Proposed Architecture (Parts 1 - 4)</div>', unsafe_allow_html=True)

# Initialize Classifier
@st.cache_resource
def load_classifier():
    return CropDiseaseClassifier(num_classes=4)

classifier = load_classifier()

# --- SIDEBAR: INPUT CONTROLS ---
st.sidebar.title("🎛️ Input Control Panel")

st.sidebar.markdown("### Part 1: Leaf Image Source")
source_mode = st.sidebar.radio("Choose Input Mode:", ["PlantVillage Benchmark Samples", "Upload Image from Device"])

selected_image_key = None
user_uploaded_image = None

if source_mode == "PlantVillage Benchmark Samples":
    selected_image_key = st.sidebar.selectbox("Select Benchmark Sample:", list(BENCHMARK_SAMPLES.keys()))
else:
    uploaded_file = st.sidebar.file_uploader("Upload Leaf Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        user_uploaded_image = Image.open(uploaded_file).convert("RGB")

st.sidebar.markdown("---")
st.sidebar.markdown("### Part 4: IoT Environmental Telemetry")
st.sidebar.caption("Simulating field sensor telemetry (ESP32 node)")

soil_moisture = st.sidebar.slider("💧 Soil Moisture (%)", min_value=10.0, max_value=95.0, value=58.0, step=1.0)
temperature   = st.sidebar.slider("🌡️ Temperature (°C)", min_value=10.0, max_value=45.0, value=28.5, step=0.5)
humidity      = st.sidebar.slider("💨 Relative Humidity (%)", min_value=20.0, max_value=100.0, value=84.0, step=1.0)
soil_ph       = st.sidebar.slider("🧪 Soil pH Level", min_value=4.0, max_value=9.0, value=6.8, step=0.1)

# Package telemetry dict
telemetry = {
    "soil_moisture": soil_moisture,
    "temperature": temperature,
    "humidity": humidity,
    "soil_ph": soil_ph
}

# --- PROCESS PIPELINE (PARTS 1 TO 4) ---
# 1. Part 1: Acquire
image_source = user_uploaded_image if user_uploaded_image is not None else selected_image_key
raw_leaf = acquire_leaf_image(image_source)

# 2. Part 2: Preprocess
resized_leaf, enhanced_leaf, tensor_input = preprocess_leaf_image(raw_leaf)

# 3. Part 3: Classify & Severity
context_hint = selected_image_key if source_mode == "PlantVillage Benchmark Samples" else None
ai_result = classify_disease_and_severity(classifier, tensor_input, context_hint=context_hint)

# 4. Part 4: Health Assessment
assessment = assess_crop_health(ai_result, telemetry)

# --- MAIN SCREEN: 3 INTERACTIVE TABS ---
tab1, tab2, tab3 = st.tabs([
    "🖼️ Part 1 & 2: Image Preprocessing",
    "🔬 Part 3: AI Classifier & Severity",
    "📊 Part 4: Multimodal Health Assessment"
])

# --- TAB 1: PREPROCESSING ---
with tab1:
    st.subheader("Architecture Part 1 & 2: Acquisition & OpenCV Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_leaf, caption="[Part 1] Raw Captured Leaf Image", use_container_width=True)
        st.info(f"**Acquisition Dimension:** {raw_leaf.shape[1]} × {raw_leaf.shape[0]} px | **Color Space:** RGB (3 channels)")

    with col2:
        st.image(enhanced_leaf, caption="[Part 2] OpenCV Enhanced (Bilateral Denoised + CLAHE)", use_container_width=True)
        st.success("**Target Resolution:** 224 × 224 px | **Filters Applied:** Edge-preserving Bilateral Filter & CLAHE Contrast Equalization")

# --- TAB 2: AI CLASSIFIER & SEVERITY ---
with tab2:
    st.subheader("Architecture Part 3: AI Disease Classification & Severity Prediction")
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown(f"""
        <div class="metric-box">
            <h4 style="margin:0; color:#1b5e20;">AI Diagnosis Result</h4>
            <h2 style="margin:5px 0; color:#2e7d32;">{ai_result['detected_class']}</h2>
            <p><b>Model Architecture:</b> MobileNetV2 Transfer Learning</p>
            <p><b>Diagnostic Confidence:</b> {ai_result['confidence']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(ai_result['confidence'] / 100.0)

    with c_right:
        tier = ai_result['severity_tier']
        box_class = "metric-box" if "Healthy" in tier else ("danger-box" if "Severe" in tier else "alert-box")
        st.markdown(f"""
        <div class="{box_class}">
            <h4 style="margin:0;">Foliar Severity Level</h4>
            <h2 style="margin:5px 0;">{tier}</h2>
            <p><b>Lesion Analysis:</b> {ai_result['severity_desc']}</p>
            <p><b>Severity Impact Index:</b> {ai_result['severity_score']:.1f} / 100.0</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: HEALTH ASSESSMENT & DOSAGE ---
with tab3:
    st.subheader("Architecture Part 4: Crop Health Assessment & Agricultural Management")
    
    col_score, col_triggers = st.columns([1, 2])
    with col_score:
        score = assessment['health_score']
        status = assessment['health_status']
        st.metric(label="Overall Crop Health Index", value=f"{score} / 100")
        
        if score >= 80.0:
            st.success(f"STATUS: {status}")
        elif score >= 50.0:
            st.warning(f"STATUS: {status}")
        else:
            st.error(f"STATUS: {status}")

    with col_triggers:
        st.markdown("<b>Active Abiotic Microclimate Risk Triggers:</b>", unsafe_allow_html=True)
        for factor in assessment['abiotic_triggers']:
            st.write(f"• {factor}")

    st.markdown("---")
    st.subheader("💊 Actionable Agronomic Prescription & Dosage (Management Engine)")
    
    rx = assessment['prescription']
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="alert-box">
            <h4 style="margin:0; color:#d84315;">Prescribed Chemical Intervention</h4>
            <p><b>Formulation:</b> {rx['Chemical Treatment']}</p>
            <p><b>Recommended Dosage:</b> {rx['Dosage']}</p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="metric-box">
            <h4 style="margin:0; color:#2e7d32;">Preventive Cultural Practices</h4>
            <p>{rx['Cultural Practice']}</p>
            <p><b>Pathology:</b> {rx['Diagnosis']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("E1CSA313 Design Project - 1 | Alliance University | AI-Based Crop Disease Detection & Management System")
