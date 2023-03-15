import pandas as pd


df_train = pd.read_csv(r"dataset\application_train.csv")
df_test = pd.read_csv(r"dataset\application_test.csv")

df_train = df_train.set_index("SK_ID_CURR")
df_test = df_test.set_index("SK_ID_CURR")
print("Data successfully imported")

key = {}
key_columns = []


for col in df_train.columns:
    # check if the column is a string
    if df_train[col].dtype == "O":
        key_columns.append(col)
        # loop over the unique strings in the dataframe
        for col_name in df_train[col].unique():
            # check if the string is not in the dictionary
            if col_name not in key:
                # add to the dictionary with a unique value
                key[col_name] = len(key) + 1
print("Key dictionary created")


# replace the string values in the dataframe with the key created
for col in key_columns:
    df_train.replace({str(col): key}, inplace=True)
for col in key_columns:
    df_test.replace({str(col): key}, inplace=True)
print("Key successfully integrated into dataframe")

try:
    df_train.to_csv(r"engineered data/df_train.csv")
    df_test.to_csv(r"engineered data/df_test.csv")
    print("Data exported to .csv")
except:
    print("Error - folder may already contain files")
