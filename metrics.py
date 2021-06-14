import numpy as np
from functools import reduce 

class Error(BaseException):
    def __init__(self, mess):
        self.message = 'Error: ' + mess

class Metrics:
    def __init__(self, y_true, y_pred, y_probs=None, task='Classification'):

        if len(y_true) != len(y_pred):
            raise Error('Different sizes y_true and y_pred')
        if self.check_types(y_true):
            raise Error('Different types in y_true list')
        if self.check_types(y_pred):
            raise Error('Different types in y_pred list')

        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_probs = np.array(y_probs)
        self.task = task

        if task == 'Classification':
            if all([elem == int or elem == np.int64 for elem in map(type, y_true)]) or all([elem == int or elem == np.int64 for elem in map(type, y_pred)]):
                self.TP, self.FP, self.TN, self.FN = self._calculate_confusion_matrix(y_true, y_pred)
            else:
                raise Error('All labels in classification task must be int type')

    def check_types(self, y_):
        types = list(map(type, y_))
        return not all(elem == types[0] for elem in types)
            
    def accuracy_score(self):
        if self.task != 'Classification':
            raise Error('Accuracy score not for regression task')
        return np.mean(self.y_true == self.y_pred)

    def precision_score(self):
        if self.task != 'Classification':
            raise Error('Precision score not for regression task')
        return self.TP / (self.TP + self.FP)

    def recall_score(self):
        if self.task != 'Classification':
            raise Error('Recall score not for regression task')
        return self.TP / (self.TP + self.FN)

    def fbeta_score(self, beta=1):
        if self.task != 'Classification':
            raise Error('Fbeta score not for regression task')
        return (1 + beta**2) / (beta**2 / self.recall_score() + 1 / self.precision_score())

    def log_loss(self, eps=1e-15):
        if self.task != 'Classification':
            raise Error('Log loss not for regression task')
        return (-1) * np.fromiter(map(lambda x, p: x * np.log(max(eps, min(1 - eps, p))) + (1 - x) * np.log(1 - max(eps, min(1 - eps, p))),\
                                      self.y_true, self.y_probs),\
                                  dtype=np.float).mean()

    def roc_auc_score(self):
        if self.task != 'Classification':
            raise Error('ROC AUC score not for regression task')
        trashholds = np.flip(np.arange(0, 1.1, 0.1))
        x_coords, y_coords = self._calculate_tpr_fpr(self.y_true, self.y_probs, trashholds)
        return np.trapz(y_coords, x_coords)

    def mean_absolute_error(self):
        if self.task == 'Classification':
            raise Error('Mean absolute error not for classification task')
        return (np.absolute(self.y_true - self.y_pred)).mean()

    def mean_squared_error(self, squared=True):
        if self.task == 'Classification':
            raise Error('Mean squared error not for classification task')
        result = (np.square(self.y_true - self.y_pred)).mean()
        return result if squared else np.sqrt(result)

    def r2_score(self):
        if self.task == 'Classification':
            raise Error('R2 score not for classification task')
        mean_ = self.y_true.mean()
        return 1 - (np.square(self.y_true - self.y_pred)).sum() / (np.square(self.y_true - mean_)).sum()
    
    def _calculate_confusion_matrix(self, y_true, y_pred):
        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(len(y_true)): 
            if y_true[i] == y_pred[i] == 1:
                TP += 1
            elif y_pred[i] == 1 and y_true[i] != y_pred[i]:
                FP += 1
            elif y_true[i] == y_pred[i] == 0:
                TN += 1
            elif y_pred[i] == 0 and y_true[i] != y_pred[i]:
                FN += 1

        return TP, FP, TN, FN

    def _calculate_tpr_fpr(self, y_true, y_probs, trashholds):
        x_coords, y_coords = np.array([]), ([])
        for trashhold in trashholds:
            y_pred = [1 if elem > trashhold else 0 for elem in y_probs]
            TP, FP, TN, FN = self._calculate_confusion_matrix(y_true, y_pred)
            x_coords = np.append(x_coords, FP / (FP + TN))
            y_coords = np.append(y_coords, TP / (TP + FN))

        return x_coords, y_coords
