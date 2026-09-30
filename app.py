import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image


@st.cache_resource
def charger_modele():
    return load_model("model/mnist_model.keras")


def preprocess(image):
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


model = charger_modele()

st.title("Reconnaissance de chiffres manuscrits")
st.write("Dessinez ou importez un chiffre (0-9)")

uploaded_file = st.file_uploader("Choisir une image", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Image chargée", width=150)

    processed = preprocess(image)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)

    st.subheader(f"Prédiction : {predicted_class}")
    st.bar_chart(prediction[0])