import uvicorn
import pandas as pd
import numpy as np
import keras

from fastapi import FastAPI
from form import application_form
from pipepline import preprocessing_pipline


app = FastAPI()
model = keras.models.load_model(r"../models/neural_network/BayOpt_v1.03/")


@app.post("/predict")
def predict_application(data: application_form):
    df = data.dict()
    df = pd.DataFrame(df, index=[0])

    scaled_data = preprocessing_pipline(df)

    y_pred = model.predict([scaled_data])
    # print(y_pred)

    return {"prediction": float(y_pred)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn app:app --reload
