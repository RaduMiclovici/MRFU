import streamlit as st
from predict import predict
from PIL import Image

st.title("Cat & Dog Breed Recognition")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    img.save("temp.jpg")

    if st.button("Predict"):
        label, conf = predict("temp.jpg")
        st.write("Prediction:", label)
        st.write("Confidence:", f"{conf*100:.2f}%")
