import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Energy Load Predictor", layout="centered")

st.title("🏠 Energy Efficiency Prediction")
st.markdown("Predict **Heating Load** or **Cooling Load** using baseline or tuned models.")

# Sidebar for model and target selection
st.sidebar.header("Model Configuration")
target = st.sidebar.radio("Select Prediction Target:", ["Heating", "Cooling"])
model_type = st.sidebar.selectbox("Select Model Version:", ["Baseline", "Tuned"])

# Build correct filename
model_filename = f"{model_type.lower()}_{target.lower()}.pkl"

# Load model
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"Model file `{model_filename}` not found.")
    st.stop()

# Input fields with realistic ranges based on your example
st.subheader("🔢 Enter Building Parameters")
col1, col2 = st.columns(2)

with col1:
    rc = st.slider("Relative Compactness", min_value=0.5, max_value=1.0, value=0.76, step=0.01)
    sa = st.slider("Surface Area", min_value=400.0, max_value=850.0, value=514.5, step=1.0)
    wa = st.slider("Wall Area", min_value=200.0, max_value=400.0, value=294.0, step=1.0)
    ra = st.slider("Roof Area", min_value=100.0, max_value=300.0, value=110.25, step=1.0)

with col2:
    oh = st.selectbox("Overall Height", options=[3.5, 7.0], index=1)  # 7.0 is from your sample
    orientation = st.selectbox("Orientation", options=[2, 3, 4, 5])  # match actual encoded range (if 1-4, use that)
    glazing_area = st.slider("Glazing Area", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
    glazing_dist = st.selectbox("Glazing Area Distribution", options=[0, 1, 2, 3, 4, 5], index=0)


# Prediction
features = np.array([[rc, sa, wa, ra, oh, orientation, glazing_area, glazing_dist]])

if st.button("🔍 Predict"):
    prediction = model.predict(features)
    st.success(f"Predicted {target} Load: **{prediction[0]:.2f}**")

st.markdown("""---""")
st.caption("💡 Make sure all model files are in the same directory as this app.")
