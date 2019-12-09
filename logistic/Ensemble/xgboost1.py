from collections import defaultdict
from collections import OrderedDict
from IPython import embed
from tqdm import tqdm
# import spacy
# nlp = spacy.load('en_core_web_md')
from data import *
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import fasttext

class XgboostClassifier():
     def __init__(self, max_depth = 3, eta = 0.3, nthread = 4, num_round = 100):
        self.max_depth = max_depth
        self.eta = eta
        self.nthread = nthread
        self.num_round = num_round
        self.bst = None

     def fit(self, x_train, y_train):
        x_train_np = np.asarray(x_train)
        y_train_np = np.asarray(y_train)
        dtrain = xgb.DMatrix(x_train_np, label=y_train_np)
        self.bst = xgb.train(self.get_params(), dtrain, self.num_round)

     def predict(self, X):
        np_sentence = np.asarray(X)
        dtest = xgb.DMatrix(np_sentence)
        labels = bst.predict(dtest)
        return labels

     def predict_proba(self, X):
        np_sentence = np.asarray(X)
        dtest = xgb.DMatrix(np_sentence)
        labels = self.bst.predict(dtest)
        return np.array([labels, 1-labels]).T
    
     def get_params(self, deep=True):
        return {"max_depth": self.max_depth, "eta": self.eta, "nthread": self.nthread, "num_round": self. num_round}