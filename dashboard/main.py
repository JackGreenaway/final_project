import streamlit as st
import requests
import numpy as np
import pandas as pd
import random


data = pd.read_csv("../data/processed_data/complete_data.csv", index_col=0)
data.drop(["TARGET"], axis=1, inplace=True)


# functions
# selects a random row from the test dataset
def select_random_data(data):
    index = random.randint(0, data.shape[0])
    random_input = data[index : index + 1]
    print(f"Index: {random_input.index[0]}")
    random_input = random_input.to_dict(orient="records")

    return random_input[0]

# accesses FastAPI to get predictions
def generate_pred():
    url = "http://127.0.0.1:8000/predict"

    rand_data = select_random_data(data)

    response = requests.post(url, json=rand_data)
    response = response.json()
    
    st.write(f"Prediction: {response['prediction']}")
    st.write("\nInputted features:")
    st.json(rand_data)


# sidebar
st.sidebar.title("")
st.sidebar.button("Generate prediction", on_click=generate_pred)
st.sidebar.write("This button randomly selects a index within the test dataset to test the model on")

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")

st.sidebar.write("Note the FastAPI also needs to be running locally for the dashboard to work")
st.sidebar.write("The predictions are made using a neural network")
st.sidebar.write("More informaiton concering the project can be found on my GitHub")
st.sidebar.write("https://github.com/JackGreenaway/final_project/")
