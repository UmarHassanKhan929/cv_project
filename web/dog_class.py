import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io
import pandas as pd

st.set_option("deprecation.showfileUploaderEncoding", False)

# Load the model
model = tf.keras.models.load_model("dog_breeds_model.h5", compile=False)

# Read the CSV file
dog_breeds = pd.read_csv('dog_breeds.csv')

# Create a dictionary mapping breed names to breed indices
breed_index_dict = dict(zip(dog_breeds['breed'], dog_breeds.index))

def preprocess_image(image, target_size):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img = img_to_array(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@st.cache(allow_output_mutation=True)
def predict_dog_breed(image):
    img = preprocess_image(image, target_size=(299, 299))

    # Make predictions
    prediction = model.predict(img)

    # Find the index of the breed with the highest probability
    breed_index = np.argmax(prediction)

    # Find the breed name corresponding to the index
    breed_name = dog_breeds.loc[breed_index]['breed']

    return breed_name


st.title("Dog Breed Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        breed = predict_dog_breed(uploaded_file)
        breed = breed.split('-')[1]
        st.write(f"Predicted breed: {breed}")
