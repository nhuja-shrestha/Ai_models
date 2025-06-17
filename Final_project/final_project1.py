import streamlit as st
import numpy as np
import joblib

# Load models
@st.cache_resource
def load_models():
    baseline = joblib.load("fraud_baseline_model.pkl")
    tuned = joblib.load("fraud_tuned_model.pkl")
    return baseline, tuned

baseline_model, tuned_model = load_models()

# Input feature list
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Proper default values (all as float, not list)
default_values = {
    'Time': 50000.0,
    'Amount': 100.0,
}
for v in [f'V{i}' for i in range(1, 29)]:
    default_values[v] = 0.0  # set PCA components to 0

st.title("💳 Credit Card Fraud Detection App")

# Model selector
model_choice = st.selectbox("Select Model", ["Baseline Model", "Tuned Model"])
model = baseline_model if model_choice == "Baseline Model" else tuned_model

st.markdown("### Enter transaction details below:")
input_data = []

# Two columns for layout
left, right = st.columns(2)

for i, feature in enumerate(feature_names):
    col = left if i % 2 == 0 else right
    with col:
        if feature == "Amount":
            value = st.slider(feature, min_value=0.0, max_value=5000.0, value=float(default_values[feature]), step=1.0)
        elif feature == "Time":
            value = st.slider(feature, min_value=0.0, max_value=172800.0, value=float(default_values[feature]), step=100.0)
        else:
            value = st.slider(feature, min_value=-30.0, max_value=30.0, value=float(default_values[feature]), step=0.1)
        input_data.append(value)

# Prediction
if st.button("🔍 Predict Fraud?"):
    arr = np.array(input_data).reshape(1, -1)
    prediction = model.predict(arr)[0]
    probability = model.predict_proba(arr)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - probability:.2%})")
