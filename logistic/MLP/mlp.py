from collections import defaultdict
from collections import OrderedDict
from IPython import embed
from tqdm import tqdm
from data import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
import itertools
from itertools import chain
import torch
from myfile import *
import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Flatten
from keras import optimizers
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pylab as plt

from visualization import attention_visualization


def main():
    mlp_obj = Logistic()
    corpus = Corpus.get_corpus("../../Data/formatted/", "./corpus.io.pkl")
    x_train, y_train, xy_train, pos_tags = mlp_obj.generate_features('./bio_probs_train.txt')

    x_test, y_test, xy_test, _ = mlp_obj.generate_features('./bio_probs_test.txt', pos_tags)
    x_val, y_val, xy_val, _ = mlp_obj.generate_features('./bio_probs_val.txt', pos_tags)
    # print(len(x_train[0]))
    # mlp = MLPClassifier(activation='relu', solver = 'adam', alpha = 1e-5, hidden_layer_sizes=(5,5,), max_iter=1000, batch_size=32)
    # mlp.fit(x_train, y_train)

    model = Sequential()
    x_train_array = np.asarray(x_train)
    y_train_array = np.asarray(y_train)
    
    x_val_array = np.array(x_val)
    y_val_array = np.array(y_val)
    model.add(Dense(units=64, activation='relu', input_dim=x_train_array.shape[1]))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=1, activation="sigmoid"))
    optimizer = keras.optimizers.Adam(lr=1e-3)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['acc'])
    history = model.fit(
          x_train_array,
          y_train_array,
          epochs=10,
          callbacks=callbacks,
          validation_data=(x_val_array, y_val_array),
          verbose=2)
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    y_pred = []
    y_test_roc = []
    y_pred_roc = []
    y_test_prob = []
    # #stores sentence features, sentence_probs, word_dict respectively
    x_test, y_test, word_dict = xy_test
    sent_len = []
    for i in x_test:
        sent_len.append(len(x_test[i]))
    myset = set(sent_len)
    new_sent_len = list(myset)
    len_map = {key:[] for key in new_sent_len}
    mean_map = {key:[] for key in new_sent_len}
    
    for i in x_test:
        x_test_array = np.array(x_test[i])
        labels = model.predict_proba(x_test_array)
        pred_label = [1 if item >= 0.5 else 0 for item in labels]
        true_label = [1 if item >= 0.5 else 0 for item in y_test[i]]
        acc_score = accuracy_score(true_label,pred_label)
        l = len(x_test[i])
        len_map[l].append(acc_score)
        l_new = []
        for j in range(len(labels)):
            
            l = labels[j].tolist()
            l_new.extend(l)
        y_pred.append(l_new)
        y_pred_roc.extend(labels)
        y_test_prob.append(y_test[i])
    embed()
    text_flat = []
    for test in corpus.test.words:
        text_flat.append(" ".join(test))

    attention_visualization.createHTML(text_flat, y_pred, "res/mlp.html")
    y_pred_roc = torch.tensor(y_pred_roc)
    _, pred = torch.max(y_pred_roc, 1)
    y_test_roc = []
    for i in y_test_prob:
        y_test_roc.extend(i)
    y_test_roc = np.array([1 if i >= 0.5 else 0 for i in y_test_roc])
    pred = pred.numpy()
    sum_all = 0
    for k,v in len_map.items():
        sum_all += len(v)
        mean_map[k] = sum(v)/len(v)

    list_sorted = sorted(mean_map.items(), key=lambda x:x[0])
    x_list = []
    y_list = []
    for i in list_sorted:
        # print(i[0],":",i[1])
        x_list.append(i[0])
        y_list.append(i[1])
    # print("x axis:",x_list)
    # print("y_axis:",y_list)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax.plot(x_list,y_list)
    ax.set_xlabel("Sentence length",fontsize=16)
    ax.set_ylabel("Accuracy Score",fontsize=16)
    ax.set_title("Sentence length vs. Accuracy",fontsize=16)
    plt.savefig('Accuracy_Result.png')

    # plt.show()


    print('ROC: ',roc_auc_score(y_test_roc, pred))
    match_score,k_score = mlp_obj.predict_score(y_pred,y_test_prob)
    
if __name__ == "__main__":
    main()