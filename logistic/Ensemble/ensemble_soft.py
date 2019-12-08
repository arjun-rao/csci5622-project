from collections import defaultdict
from collections import OrderedDict
from IPython import embed
from tqdm import tqdm
# import spacy
import torch
import numpy as np
# nlp = spacy.load('en_core_web_md')
from data import *
from itertools import chain
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from feature import *
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def main():
    data_obj = Dataset()

    x_train, y_train, xy_train, pos_tags = data_obj.generate_features(filename='./bio_probs_train.txt')
    x_test, y_test, xy_test, _ = data_obj.generate_features(filename='./bio_probs_test.txt', pos_tags=pos_tags)
    log_reg = LogisticRegression(random_state=2019)
    # log_reg.fit(x_train, y_train)
    # mlp = MLPClassifier(activation='relu', alpha = 1e-5, hidden_layer_sizes=(5, 5), max_iter=1000, random_state=1)
    mlp = MLPClassifier(activation='relu', solver = 'adam', alpha = 1e-5, hidden_layer_sizes=(5,5,), max_iter=1000, batch_size=32)
    # mlp.fit(x_train, y_train)

    e_clf =  VotingClassifier(estimators=[('lr', log_reg), ('mlp', mlp)], voting='soft')
    e_clf.fit(x_train,y_train)

    y_pred = []
    y_test_prob = []
    y_test_roc = []
    y_pred_roc = []
    #x-test format: key:"1" value:[probs for each word]; key represents the sentence_#
    #stores sentence features, sentence_probs, word_dict respectively
    x_test, y_test, word_dict = xy_test
    
    for i in x_test:
        labels = e_clf.predict_proba(x_test[i])
        y_pred_roc.extend(labels)
        y_pred.append([item[1] for item in labels])
        y_test_prob.append(y_test[i])
    y_pred_roc = torch.tensor(y_pred_roc)
    _, pred = torch.max(y_pred_roc, 1)
    y_test_roc = []
    for i in y_test_prob:
        y_test_roc.extend(i)
    y_test_roc = np.array([1 if i >= 0.5 else 0 for i in y_test_roc])
    
    pred = pred.numpy()
    print('ROC: ',roc_auc_score(y_test_roc, pred))
    match_score,k_score = data_obj.predict_score(y_pred,y_test_prob)
    # embed()


if __name__ == "__main__":
    main()