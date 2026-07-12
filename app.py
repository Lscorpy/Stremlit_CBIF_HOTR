# app.py
import streamlit as st
import requests
import base64

st.title("CBIF-HOTR Violence Detection — Local Test")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input image", width=400)

    if st.button("Run inference"):
        with st.spinner("Running model..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files, timeout=120)

        if response.status_code == 200:
            data = response.json()
            record = data["record"]


            st.subheader("HOI interactions")
            st.json(record["hoi_interactions"])

            st.subheader("HHI interactions")
            st.json(record["hhi_interactions"])

            st.image(base64.b64decode(data["stage1_image_base64"]),
                    caption="Stage 1 — Detection", use_container_width=True)
            st.image(base64.b64decode(data["stage2_image_base64"]),
                    caption="Stage 2 — Roles", use_container_width=True)
            st.image(base64.b64decode(data["stage3_image_base64"]),
                    caption="Stage 3 — Triplets", use_container_width=True)
        else:
            st.error(f"Request failed: {response.status_code}")
            st.text(response.text)