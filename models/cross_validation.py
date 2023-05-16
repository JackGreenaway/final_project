import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import performance_metrics


class cross_val:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def run_validation(self, kfold, model):
        """
        This function runs all of the cross validation for the model
        
        Returns a dataframe containing the performance metrics for each model, as well as a confusion matrix
        """
        
        results = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr"])
        results.index.name = "Model"
        
        counter = 1
        confusion_matrix_sum = None
        
        for train_index, test_index in kfold.split(self.x):
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            # fit model to train set
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            # calculate confusion matrix, and append data for later visualisation
            cm = confusion_matrix(y_test, y_pred)
            fpr = cm[0][1] / cm.sum()
            fnr = cm[1][0] / cm.sum()
            
            if confusion_matrix_sum is None:
                confusion_matrix_sum = cm
            else:
                confusion_matrix_sum += cm
            
            # calculate metrics and append to dataframe
            metrics = performance_metrics.metrics(y_test, y_pred) 
            accuracy, precision, recall, f1, auc_result = metrics.calculate_metrics()
            results.loc[counter] = [accuracy, precision, recall, f1, auc_result, fpr, fnr]
            
            print(f"Fold {counter} completed")
            counter += 1

        # calculate the average over all models
        results.loc["Average"] = [results.accuracy.mean(),
                                results.precision.mean(),
                                results.recall.mean(),
                                results.f1.mean(),
                                results.auc.mean(),
                                results.fpr.mean(),
                                results.fnr.mean()]
        
        return results, confusion_matrix_sum