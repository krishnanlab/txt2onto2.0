from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod
from sklearn.metrics import average_precision_score as auPRC
from sklearn.metrics import f1_score as f1
import numpy as np


class Model(ABC):
    '''Abstract base class for models'''
    @abstractmethod
    def fit(self, X, y):
        '''Abstract method to fit the model'''
        pass

    @abstractmethod
    def predict(self, X):
        '''Abstract method for making prediction'''
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predict):
        '''Abstract method for evaluating the model'''
        pass

    @abstractmethod
    def get_coef(self):
        '''Abstract method for getting the model coefficients'''
        pass


class LogisticRegressionModel(Model):
    ''' Logistic Regression model class'''
    def __init__(self, C=1, l1r=1):
        '''
        Initialize the Logistic Regression model.
        C: Regularization strength (default=1)
        l1r: L1 ratio for elastic net regularization (default=1)
        '''
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
        '''
        Fit the Logistic Regression model
        X: TF-IDF matrix
        y: labels
        '''
        self.model.fit(X, y)

    def predict(self, X):
        '''
        Make predictions using the Logistic Regression model
        X: TF-IDF matrix
        return: probability
        '''
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, y_true, y):
        '''
        Evaluate the Logistic Regression model using log2(auPRC/prior).
        y_true: True labels
        y: Predicted probabilities
        return: log2(auPRC/prior) score
        '''
        return np.log2(auPRC(y_true, y)/np.mean(y_true))

    def get_coef(self):
        '''
        Get the coefficients of the Logistic Regression model.
        return: Model coefficients
        '''
        return self.model.coef_[0]


class PowerMatrix(Model):
    '''Power Matrix model class'''
    def __init__(self, metric='auprc'):
        '''
        Initialize the Power Matrix model.
        metric: Metric to use for calculating scores for each feature (default='auprc')
        '''
        self.metric = metric
        self.beta = list()

    def fit(self, X, y):
        '''
        Calculate metric for every training data
        X: TF-IDF matrix
        y: labels
        '''
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
        '''
        Make predictions using the Power Matrix model.
        X: TF-IDF matrix
        return: Predicted scores, which is average of metric weighted by TF-IDF
        '''
        betas = np.array(self.beta)
        word_scores = np.multiply(X / np.sum(X, axis=1)[:, None], betas)

        return np.sum(word_scores, axis=1)

    def evaluate(self, y_true, probs):
        '''
        Evaluate the Power matrix model using log2(auPRC/prior).
        y_true: True labels
        y: Predicted probabilities
        return: log2(auPRC/prior) score
        '''
        model_performance = np.log2(auPRC(y_true, np.array(probs))/np.mean(y_true))
        return model_performance

    def get_coef(self):
        '''
        Get the coefficients of the Power matrix model.
        return: Model coefficients
        '''
        return np.array(self.beta)
