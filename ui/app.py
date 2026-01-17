import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Self-Healing ML System", layout="centered")

st.title("Self-Healing ML System")
st.caption("Production-style ML inference with drift detection and auto-recovery")


st.subheader("Input Features")

features = []
cols = st.columns(5)

for i in range(5):
    with cols[i]:
        features.append(st.number_input(f"f{i}", value=0.5))


if st.button("Run Prediction"):
    payload = {"features": [features]}

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
    except Exception as e:
        st.error("Inference service not reachable")
        st.stop()

    if response.status_code != 200:
        st.error("Inference error")
        st.stop()

    data = response.json()

    st.subheader("Prediction Result")
    prediction = float(data["predictions"][0][0])
    st.success(f"Prediction value: **{prediction:.4f}**")


    st.subheader("System Health")

    if data["drift_detected"]:
        st.warning("Data drift detected")
        st.info(
            "The system has flagged abnormal input patterns.\n\n"
            "**Self-healing pipeline will automatically retrain and evaluate the model in the background.**"
        )
    else:
        st.success("System healthy — no drift detected")


st.divider()
st.subheader("System Notes")

st.markdown(
"""
- This UI **does not control** the ML system.
- All decisions (retraining, promotion, rollback) are handled automatically.
- The UI only reflects **current system state**.
"""
)
