import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    auc,
)


class metrics:
    def __init__(self, y_true, y_pred) -> None:
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        # print(f"{'-'*10} Evaluation Metrics {'-'*10}")

        accuracy = accuracy_score(self.y_true, self.y_pred)
        accuracy = "%.3f" % accuracy 
        
        precision = precision_score(self.y_true, self.y_pred)
        precision = "%.3f" % precision 

        recall = recall_score(self.y_true, self.y_pred)
        recall = "%.3f" % recall 

        f1 = f1_score(self.y_true, self.y_pred)
        f1 = "%.3f" % f1 

        auc_result = roc_auc_score(self.y_true, self.y_pred)
        auc_result = "%.3f" % auc_result

        results = pd.DataFrame(
            data=[accuracy, precision, recall, f1, auc_result],
            index=["Accuracy", "Precision", "Recall", "F1-score", "AUC-score"],
            columns=["Score"],
        )
        print(results)

    def plot_roc(self, ns_y_pred):
        # ns_y_pred = non-scaled outputs of tf model
        fpr, tpr, threshold = roc_curve(self.y_true, ns_y_pred)
        roc_auc = auc(fpr, tpr)

        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        disp.plot()

    def plot_confusion_matrix(self):
        con_matrix = confusion_matrix(self.y_true, self.y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=con_matrix, display_labels=["Default", "Non-default"]
        )
        disp.plot(cmap=plt.cm.Blues)

        # Extract true negative, false positive, false negative, true positive from the confusion matrix
        tn, fp, fn, tp = con_matrix.ravel()
        # Calculate false positive rate (FPR)
        fpr = fp / (fp + tn)
        print(f"False positive rate: {'%.2f' % (fpr*100)}%")
