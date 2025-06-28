import streamlit as st
from PIL import Image
from MLflow.app import MLflowClient
from Model.predictor import FoodClassifier

# python -m streamlit run Streamlit/app.py

st.header("Streamlit food classification app -Group 2-")
st.balloons()

st.title("SUML Food11 Classifier")
st.write("Input an image below.")

uploaded = st.file_uploader("Select JPG or PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Photo", use_container_width=True)

    classifier = FoodClassifier("Model/food11.onnx")
    label, confidence = classifier.predict(image)
    client = MLflowClient()
    label, confidence = client.predict(image=image)

    st.success(f"Predicted Food Type: **{label}** ({confidence * 100:.2f}% confidence)")
