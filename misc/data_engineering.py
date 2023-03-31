import pandas as pd

"""
sklearn.preprocessing.LabelEncoder could be used...
however, that only works with 1d arrays therefore, I feel this method is easier to use
"""

# this script takes a .csv file and changes columns with string values, creates a key dictionary
# then replaces the strings with the key dictionary and saves it to a new .csv file

# import the .csv
df_train = pd.read_csv(r"data\application_train.csv")
df_test = pd.read_csv(r"data\application_test.csv")

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
# print(key)


print("Integrating keys into dataframe...")
# replace the string values in the dataframe with the key created
for col in key_columns:
    df_train.replace({str(col): key}, inplace=True)  # train
print("Train dataframe intergrated")
for col in key_columns:
    df_test.replace({str(col): key}, inplace=True)  # test
print("Test dataframe intergrated")
print("Key successfully integrated into dataframe")

# saved the engineered data in a new .csv
try:
    df_train.to_csv(r"data/df_train.csv")
    df_test.to_csv(r"data/df_test.csv")
    print("Data exported to .csv @ .../data/")
except:
    print("Error - folder may already contain files or, check the save location")
