import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas


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


def afficher_prediction(image):
    processed = preprocess(image)
    prediction = model.predict(processed, verbose=0)
    predicted_class = np.argmax(prediction)
    st.subheader(f"Prédiction : {predicted_class}")
    st.bar_chart(prediction[0])


model = charger_modele()

st.title("Reconnaissance de chiffres manuscrits")
st.write("Dessinez ou importez un chiffre (0-9)")

mode = st.radio(
    "Mode",
    ["Importer une image", "Dessiner"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Importer une image":
    uploaded_file = st.file_uploader("Choisir une image", type=["png", "jpg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Image chargée", width=150)
        afficher_prediction(image)
else:
    st.caption("Dessine un chiffre dans le cadre, bien au centre")
    with st.container(border=True):
        canvas = st_canvas(
            stroke_width=20,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    if canvas.json_data is not None and canvas.json_data["objects"]:
        image = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
        afficher_prediction(image)