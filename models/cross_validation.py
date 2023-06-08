import pandas as pd
import numpy as np
import performance_metrics
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Filter the specific warning message
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)


class cross_val:
    def __init__(self, x, y) -> None:
        self.x = np.array(x)
        self.y = np.array(y)

    def run_validation(self, kfold, model, scaler):
        """
        This function runs all of the cross validation for the model

        Returns a dataframe containing the performance metrics for each model, as well as a confusion matrix
        """

        results = pd.DataFrame(
            columns=["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr"]
        )
        results.index.name = "Model"

        counter = 1
        confusion_matrix_sum = None

        for train_index, test_index in kfold.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # scale the x variables (fitting only to the train data)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)

            # fit model to train set
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # calculate confusion matrix, and append data for later visualisation
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[1][1]
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]

            fpr = fp / (tn + fp)
            fnr = fn / (tp + fn)

            if confusion_matrix_sum is None:
                confusion_matrix_sum = cm
            else:
                confusion_matrix_sum += cm

            # calculate metrics and append to dataframe
            metrics = performance_metrics.metrics(y_test, y_pred)
            accuracy, precision, recall, f1, auc_result = metrics.calculate_metrics()
            results.loc[counter] = [
                accuracy,
                precision,
                recall,
                f1,
                auc_result,
                fpr,
                fnr,
            ]

            print(f"Fold {counter} completed")
            counter += 1

        # calculate the average over all models
        results.loc["Average"] = [
            results.accuracy.mean(),
            results.precision.mean(),
            results.recall.mean(),
            results.f1.mean(),
            results.auc.mean(),
            results.fpr.mean(),
            results.fnr.mean(),
        ]

        print("Validation Completed")

        return results, confusion_matrix_sum
