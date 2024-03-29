import uvicorn
import pandas as pd
import numpy as np
import mlflow
import keras
from scipy import stats
from sqlalchemy import create_engine, engine

from fastapi import FastAPI
from form import application_form
from pipepline import preprocessing_pipline


app = FastAPI()
model = keras.models.load_model(r"../models/neural_network/nn_v1")
feature_names = pd.read_csv("col_names", index_col=0)
engine = create_engine("sqlite:///input_warehouse.db")

"""
This is a endpoint to make predictions from using the neural network
"""
@app.post("/predict")
def predict_application(data: application_form):
    # format the data to a df
    df_dict = data.dict()
    df = pd.DataFrame(df_dict, index=[0])
    
    # rename columns to that of the expected feature names passed during fit
    fn = list(feature_names["0"][:-1])
    df.columns = fn

    # scale are predict
    scaled_data = preprocessing_pipline(df)
    y_pred = model.predict([scaled_data])

    # add the inputs and prediction to a database
    # df["y_pred"] = y_pred
    # df.to_sql("model_inputs", engine, if_exists="append")

    # log the prediction to mlflow
    # with mlflow.start_run(run_name="home_default_classification"):
    #     mlflow.log_metric("prediction", y_pred[0])
    #     mlflow.log_params(df_dict)

    return {"prediction": float(y_pred)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn main:app --reload
