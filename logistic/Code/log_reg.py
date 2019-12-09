import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from features import *
from sklearn.svm import SVC



def gen_feature():
    features_obj = Features()
    x_train, y_train, xy_train, pos_tags = features_obj.generate_features(filename='./bio_probs_train.txt',glove_emb=False)
    x_test, y_test, xy_test, _ = features_obj.generate_features(filename='./bio_probs_test.txt', pos_tags=pos_tags, glove_emb=False) 
    return x_train, y_train, xy_train, pos_tags, x_test, y_test, xy_test,_


def log_reg():
    features_obj = Features()
    x_train, y_train, xy_train, pos_tags,x_test, y_test, xy_test,_ = gen_feature()
    
    clf = LogisticRegression(random_state=2019)
    clf.fit(x_train, y_train)
    y_pred = []
    y_test_prob = []
    y_test_roc = []
    y_pred_roc = []
    #x-test format: key:"1" value:[probs for each word]; key represents the sentence_#
    #stores sentence features, sentence_probs, word_dict respectively
    x_test, y_test, word_dict = xy_test
    
    for i in x_test:
        labels = clf.predict_proba(x_test[i])
        y_pred.append([item[1] for item in labels])
        y_test_prob.append(y_test[i])
        y_pred_roc.extend(labels)
    y_pred_roc = torch.tensor(y_pred_roc)
    _, pred = torch.max(y_pred_roc, 1)
    for i in y_test_prob:
        y_test_roc.extend(i)
    y_test_roc = np.array([1 if i >= 0.5 else 0 for i in y_test_roc])
    pred = pred.numpy()
    print('ROC: ',roc_auc_score(y_test_roc, pred))
    match_score,k_score = features_obj.predict_score(y_pred,y_test_prob)


def svm_lin():
    # clf = LinearSVC(random_state=0, tol=1e-5)

    features_obj = Features()
    x_train, y_train, xy_train, pos_tags,x_test, y_test, xy_test,_ = gen_feature()
    print("Finished")
    clf = SVC(random_state=2019, probability=True)
    print("Finished1")
    clf.fit(x_train, y_train)
    print("Finished2")
    y_pred = []
    y_test_prob = []
    y_test_roc = []
    y_pred_roc = []
    #x-test format: key:"1" value:[probs for each word]; key represents the sentence_#
    #stores sentence features, sentence_probs, word_dict respectively
    x_test, y_test, word_dict = xy_test
    
    for i in x_test:
        labels = clf.predict_proba(x_test[i])
        y_pred.append([item[1] for item in labels])
        y_test_prob.append(y_test[i])
        y_pred_roc.extend(labels)
    print("Finished3")
    y_pred_roc = torch.tensor(y_pred_roc)
    _, pred = torch.max(y_pred_roc, 1)
    for i in y_test_prob:
        y_test_roc.extend(i)
    print("Finished4")
    y_test_roc = np.array([1 if i >= 0.5 else 0 for i in y_test_roc])
    pred = pred.numpy()
    print('ROC: ',roc_auc_score(y_test_roc, pred))
    print("Finished5")
    match_score,k_score = features_obj.predict_score(y_pred,y_test_prob)
    

def main():
    # log_reg()
    svm_lin()    



if __name__ == "__main__":
    main()