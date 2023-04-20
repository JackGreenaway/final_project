from sklearn.preprocessing import StandardScaler
from pickle import load


def preprocessing_pipline(data):
    scaler = load(open(r"../models/scaler.pkl", "rb"))
    scaled_data = scaler.transform(data)

    return scaled_data
