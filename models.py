import numpy as np


class Metrics:
    cm = None
    values = {}

    @staticmethod
    def set_confusion_matrix(confusion_matrix):
        Metrics.cm = confusion_matrix

    @staticmethod
    def set_values(num_value):
        Metrics.values = Metrics._get_cm_values(num_value)

    @staticmethod
    def _get_cm_values(num_value):
        cm = Metrics.cm
        if isinstance(cm, np.ndarray):
            values = {
                "TP": cm[0, 0],
                "FN": cm[0, 1] + cm[0, 2],
                "FP": cm[1, 0] + cm[2, 0],
                "TN": cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2],
            }

            if num_value == 1:
                values["TP"] = cm[1, 1]
                values["FN"] = cm[1, 0] + cm[1, 2]
                values["FP"] = cm[0, 1] + cm[2, 1]
                values["TN"] = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]
            elif num_value == 2:
                values["TP"] = cm[1, 1]
                values["FN"] = cm[2, 0] + cm[2, 1]
                values["FP"] = cm[0, 0] + cm[0, 1]
                values["TN"] = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]

            return values

        return {}

    def __init__(self):
        self.accuracy = []
        self.precision = []
        self.sensitivity = []
        self.f1 = []

    def add_accuracy(self):
        values = Metrics.values
        self.accuracy.append(
            100
            * sum([values["TP"], values["TN"]])
            / sum([values["TP"], values["TN"], values["FP"], values["FN"]])
        )

    def add_precision(self):
        values = Metrics.values
        self.precision.append(100 * values["TP"] / (values["TP"] + values["FP"]))

    def add_sensitivity(self):
        values = Metrics.values
        self.sensitivity.append(100 * values["TP"] / (values["TP"] + values["FN"]))

    def add_f1(self):
        values = Metrics.values
        self.f1.append(
            100
            * (2 * values["TP"])
            / (2 * sum([values["TP"], values["FP"], values["FN"]]))
        )

    def set_metrics(self):
        self.add_accuracy()
        self.add_precision()
        self.add_sensitivity()
        self.add_f1()
