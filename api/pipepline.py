from sklearn.preprocessing import StandardScaler
from pickle import load


def preprocessing_pipline(data):
    scaler = load(open(r"C:\Users\jack_\OneDrive - University of Gloucestershire\Financial Technology (2022-2023)\final_project\models\scaler.pkl", "rb"))
    scaled_data = scaler.transform(data)

    return scaled_data
