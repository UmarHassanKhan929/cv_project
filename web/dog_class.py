import streamlit as st
import openai
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import requests

openai.api_key = "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"

st.set_option("deprecation.showfileUploaderEncoding", False)

model = tf.keras.models.load_model("dog_breeds_model.h5", compile=False)

dog_breeds = pd.read_csv('dog_breeds.csv')

# Create a dictionary mapping breed names to breed indices
breed_index_dict = dict(zip(dog_breeds['breed'], dog_breeds.index))


def img_to_array(image):
    img = np.array(image)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    img = img.transpose(2, 0, 1)
    return img


def preprocess_image(image, target_size):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img = img_to_array(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0, 2, 3, 1))
    return img


@st.cache(allow_output_mutation=True)
def predict_dog_breed(images):
    predictions = []
    for image in images:
        img = preprocess_image(image, target_size=(299, 299))

        # Make predictions
        prediction = model.predict(img)

        # Find the index of the breed with the highest probability
        breed_index = np.argmax(prediction)

        # Find the breed name corresponding to the index
        breed_name = dog_breeds.loc[breed_index]['breed']
        predictions.append(breed_name)

        for idx in prediction.argsort()[0][::-1][:5]:
            print("{:.2f}%".format(
                prediction[0][idx]*100), "\t", dog_breeds.loc[idx]['breed'].split("-")[-1])

    return max(set(predictions), key=predictions.count)


def generate_text(prompt):
    reqUrl = "https://api.openai.com/v1/completions"
    reqHeaders = {
        'Accept': 'text/event-stream',
        'Authorization': 'Bearer ' + openai.api_key,
    }
    reqBody = {
        "model": "text-davinci-003",
        "prompt": f"Give me 3 fun fact about ${prompt}",
        "max_tokens": 180,
        "temperature": 0.5,
    }
    request = requests.post(reqUrl, headers=reqHeaders, json=reqBody)
    response = request.json()
    print(response)
    return response['choices'][0]['text']


def getMoreInfo(breed, prompt):
    print(breed, prompt)
    # reqUrl = "https://api.openai.com/v1/completions"
    # reqHeaders = {
    #     'Accept': 'text/event-stream',
    #     'Authorization': 'Bearer ' + openai.api_key,
    # }
    # reqBody = {
    #     "model": "text-davinci-003",
    #     "prompt": f"Tell me more about the {breed}'s  ${prompt}",
    #     "max_tokens": 60,
    #     "temperature": 0.5,
    # }
    # request = requests.post(reqUrl, headers=reqHeaders, json=reqBody)
    # response = request.json()
    # print(response)
    # st.write(f"Breed:{breed}\n"+response['choices'][0]['text'])


# Main app
st.title("Dog Breed Classifier")

uploaded_files = st.file_uploader(
    "Choose an image...", type=["jpg", "png"], accept_multiple_files=True)

if "predictedBreed" not in st.session_state:
    st.session_state.predictedBreed = ""

if uploaded_files is not None:
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        st.image(
            image, caption=f"Uploaded Image: {i+1}/{len(uploaded_files)}", use_column_width=True)

    if st.button("Predict"):
        breed = predict_dog_breed(uploaded_files)
        breed = breed.split('-')[1]
        st.session_state.predictedBreed = breed
        st.write(f"Predicted breed: {breed}")

        st.write("Here's a fun fact about your dog breed:")
        st.write(generate_text(breed))
        st.write(breed)

    button_group = st.radio(
        "What do you want to learn next?", ["Nothing",
                                            "Favorite Food?", "The Dog's Personality?", "Does it behave well with strangers?"])
    if button_group == "Favorite Food?":
        getMoreInfo(st.session_state.predictedBreed, "favorite food")
    elif button_group == "The Dog's Personality?":
        getMoreInfo(st.session_state.predictedBreed, "personality")
    elif button_group == "Does it behave well with strangers?":
        getMoreInfo(st.session_state.predictedBreed, "behavior with strangers")

        # if st.button("Give me another fun fact"):
        #     # st.write(generate_text(f"{breed}"))
        #     st.write("text2")
