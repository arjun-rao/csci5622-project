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


def main():
    mlp_obj = Logistic()

    x_train, y_train, xy_train, pos_tags = mlp_obj.generate_features('./bio_probs_train.txt')

    x_test, y_test, xy_test, _ = mlp_obj.generate_features('./bio_probs_test.txt', pos_tags)
    x_val, y_val, xy_val, _ = mlp_obj.generate_features('./bio_probs_val.txt', pos_tags)
    # print(len(x_train[0]))
    # mlp = MLPClassifier(activation='relu', solver = 'adam', alpha = 1e-5, hidden_layer_sizes=(5,5,), max_iter=1000, batch_size=32)
    # mlp.fit(x_train, y_train)

    model = Sequential()
    # print("x type:",x_train[0])
    # print("y type:",type(y_train))
    # print("x_train:",x_train)
    # print("y_train:",y_train)
    # # print(type(np.asarray(x_train)))
    x_train_array = np.asarray(x_train)
    y_train_array = np.asarray(y_train)
    
    x_val_array = np.array(x_val)
    y_val_array = np.array(y_val)
    # print("Shape:",x_train_array.shape)
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
    # embed()
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    y_pred = []
    y_test_roc = []
    y_pred_roc = []
    y_test_prob = []
    # #stores sentence features, sentence_probs, word_dict respectively
    x_test, y_test, word_dict = xy_test
    
    for i in x_test:
        x_test_array = np.array(x_test[i])
        labels = model.predict_proba(x_test_array)
        l_new = []
        for j in range(len(labels)):
            
            l = labels[j].tolist()
            l_new.extend(l)
        y_pred.append(l_new)
        y_pred_roc.extend(labels)
        y_test_prob.append(y_test[i])
    y_pred_roc = torch.tensor(y_pred_roc)
    _, pred = torch.max(y_pred_roc, 1)
    # print("y_pred:",y_pred)
    # print("y test prob:",y_test_prob)
    y_test_roc = []
    for i in y_test_prob:
        y_test_roc.extend(i)
    y_test_roc = np.array([1 if i >= 0.5 else 0 for i in y_test_roc])
    pred = pred.numpy()
    print('ROC: ',roc_auc_score(y_test_roc, pred))
    match_score,k_score = mlp_obj.predict_score(y_pred,y_test_prob)
    # # embed()

if __name__ == "__main__":
    main()