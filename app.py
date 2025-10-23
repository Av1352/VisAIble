import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from generate_explanations import load_model, apply_gradcam, apply_lime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VisAIble Deepfake Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MATCH ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #20639b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #20639b;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 class="main-header">🔍 VisAIble: Explainable Deepfake Detection AI</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;margin-bottom:2rem;">
    <p style="font-size:1.2rem;color:#666;">
        AI-powered deepfake face analysis with explainable visualizations.<br>
        EfficientNet-B0 · GradCAM · LIME · Fast, interpretable, and shareable results for your portfolio.
    </p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: METRICS AND INFO ---
with st.sidebar:
    st.header("📊 Model Performance")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Accuracy", "97.75%", " ")
    with colB:
        st.metric("AUC Score", "0.99", " ")
        
    st.markdown("---")
    
    st.subheader("🧠 Model Details")
    st.write("**Architecture:** EfficientNet-B0")
    st.write("**Input Size:** 224×224")
    st.write("**XAI:** GradCAM, LIME")
    st.write("**Training Images:** 140,000+")
    
    st.markdown("---")
    
    st.subheader("📈 Model Comparison")
    comp_data = {
        'Model': ['EfficientNet', 'Other Models'],
        'Accuracy': [0.977, 0.94],
        'AUC': [0.99, 0.97]
    }
    fig_comp = px.bar(comp_data, x='Model', y='Accuracy', 
                    title='Model Performance Comparison',
                    color='Accuracy', color_continuous_scale='blues')
    fig_comp.update_layout(height=300)
    st.plotly_chart(fig_comp, use_container_width=True)

# --- MAIN CONTENT ---
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📤 Upload Face Image")
    uploaded_file = st.file_uploader(
        "Choose a face image (JPG, PNG, 224x224 recommended)",
        type=["png", "jpg", "jpeg"],
        help="Upload a face image for deepfake analysis"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write(f"**Image Size:** {image.size}")
        st.write(f"**Format:** {image.format}")
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                os.makedirs("explanations", exist_ok=True)
                img_path = "explanations/uploaded.jpg"
                image.save(img_path)

                # Load model once (cache!)
                @st.cache_resource
                def get_model():
                    return load_model()
                model = get_model()

                gradcam_path = "explanations/gradcam_uploaded.png"
                lime_path = "explanations/lime_uploaded.png"

                pred_cam = apply_gradcam(img_path, model, gradcam_path)
                pred_lime = apply_lime(img_path, model, lime_path)
                st.session_state["pred_cam"] = pred_cam
                st.session_state["pred_lime"] = pred_lime
                st.session_state["image"] = image

with col2:
    st.subheader("🎯 Analysis Results")
    if (
        "pred_cam" in st.session_state and
        "pred_lime" in st.session_state and
        "image" in st.session_state
    ):
        pred_cam = st.session_state["pred_cam"]
        pred_lime = st.session_state["pred_lime"]
        image = st.session_state["image"]

        # Prediction cards
        label = lambda x: "Real" if x==0 else "Fake"
        if pred_cam == 1 or pred_lime == 1:
            st.error(f"🔴 **DEEPFAKE DETECTED**")
            st.write(f"**Prediction:** Fake")
            prob_color = "red"
        else:
            st.success(f"✅ **REAL FACE DETECTED**")
            st.write(f"**Prediction:** Real")
            prob_color = "green"

        # Explanation visualizations
        st.subheader("🔍 Explainable AI Visualizations")
        tab1, tab2 = st.tabs(["GradCAM", "LIME"])
        with tab1:
            st.image("explanations/gradcam_uploaded.png", caption="GradCAM Heatmap", use_container_width=True)
            st.caption(f"GradCAM infers: {label(pred_cam)}")
            st.download_button(
                "Download GradCAM image",
                data=open("explanations/gradcam_uploaded.png","rb"),
                file_name="gradcam.png"
            )
        with tab2:
            st.image("explanations/lime_uploaded.png", caption="LIME Mask", use_container_width=True)
            st.caption(f"LIME infers: {label(pred_lime)}")
            st.download_button(
                "Download LIME image",
                data=open("explanations/lime_uploaded.png","rb"),
                file_name="lime.png"
            )
    else:
        st.info("👆 Upload a face image and click 'Analyze' to see results.")

# --- WARNING & TECHNICAL DETAILS ---

st.markdown("---")
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p>This is an educational demo and research project. <strong>This app is NOT for clinical use, identity authentication, or any security application.</strong> Results are for illustrative purposes only.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("🔧 Technical Details"):
    st.markdown("""
    ### Model Architecture
    - **Base Model:** EfficientNet-B0 (PyTorch, timm)
    - **Explainability:** GradCAM (spatial attention), LIME (local pixel importance)
    - **Input Shape:** 224×224×3 (RGB face images)
    - **Output:** Binary classification (real/fake)
    
    ### Training Details
    - **Dataset:** Public deepfake & real face datasets (Kaggle)
    - **Training Images:** 140,000+
    
    ### Performance Metrics
    - **EfficientNet-B0:** 97.75% accuracy, 0.99 AUC
   
    ### Technology Stack
    - **Framework:** PyTorch/timm
    - **Frontend:** Streamlit
    - **Visualization:** Matplotlib, Plotly
    - **Deployment:** Streamlit Cloud
    """)

st.markdown("---")
st.caption("Made by Anju Vilashni Nandhakumar | Powered by AI | [GitHub](https://github.com/Av1352/VisAIble) | [Portfolio](https://vxanju.com)")

