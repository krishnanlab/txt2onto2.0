from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod
from sklearn.metrics import average_precision_score as auPRC
from sklearn.metrics import f1_score as f1
import numpy as np


class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predict):
        pass

    @abstractmethod
    def get_coef(self):
        pass


class LogisticRegressionModel(Model):
    def __init__(self, C=1, l1r=1):
        self.C = C
        self.l1r = l1r
        if self.l1r == 1:
            self.model = LogisticRegression(penalty="l1",
                                            C=self.C,
                                            solver='liblinear')
        elif self.l1r == 0:
            self.model = LogisticRegression(penalty="l2",
                                            C=self.C,
                                            solver='liblinear')
        else:
            self.model = LogisticRegression(penalty="elasticnet",
                                            C=self.C,
                                            l1_ratio=l1r,
                                            solver='saga')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, y_true, y):
        return np.log2(auPRC(y_true, y)/np.mean(y_true))

    def get_coef(self):
        return self.model.coef_[0]


class PowerMatrix(Model):
    def __init__(self, metric='F1'):
        self.metric = metric
        self.beta = list()

    def fit(self, X, y):
        for word_col in range(0, X.shape[1]):
            y_pred = X.T[word_col]
            if self.metric == 'F1':
                score = f1(y_true=y, y_pred=(y_pred/y_pred.max()).round())
                self.beta.append(score)
            elif self.metric == 'auprc':
                score = np.log2(auPRC(y_true=y, y_score=y_pred)/np.mean(y))
                if score <= 0:
                    score = 0
                self.beta.append(score)

    def predict(self, X):
        betas = np.array(self.beta)
        word_scores = np.multiply(X / np.sum(X, axis=1)[:, None], betas)

        return np.sum(word_scores, axis=1)

    def evaluate(self, y_true, probs):
        model_performance = np.log2(auPRC(y_true, np.array(probs))/np.mean(y_true))
        return model_performance

    def get_coef(self):
        return np.array(self.beta)
