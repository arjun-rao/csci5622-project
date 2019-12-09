from sklearn.metrics import f1_score
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import f1_score
from itertools import chain
from sklearn.metrics import roc_auc_score
import itertools
from sklearn_crfsuite import metrics
import pickle
import os
from tqdm import tqdm, trange
from IPython import embed
import argparse

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification
from transformers import AdamW, WarmupLinearSchedule

from seqeval.metrics import f1_score

from logger import Logger
from helper import Helper
from config import *
import config
from dataset import *
from model import BertAttnModel
import attention_visualization

helper = Helper()
logger = Logger(config.output_dir_path + 'logs')

def tensor_logging(model, info, epoch):
    for tag, value in info.items():
        logger.log_scalar(tag, value, epoch + 1)
    # Log values and gradients of the model parameters
    for tag, value in model.named_parameters():
        if value.grad is not None:
            tag = tag.replace('.', '/')
            if torch.cuda.is_available():
                logger.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)


# Config parameters
# Max sequence length
MAX_LEN = 75
# batch size
bs = 32
# Number of labels
NUM_LABELS = 2

# Store viz results
fileVizMap = {
    'scores': {'text': [], 'wts': []},
    'target': {'text': [], 'wts': []},
    'target_softmax': {'text': [], 'wts': []}
}

def visualize_attention(wts,words, filename, maps, viz_type='scores'):
    """
    Visualization function to create heat maps for prediction, ground truth and attention (if any) probabilities
    :param wts:
    :param words:
    :param filename:
    :return:
    """
    wts_add = wts.cpu()
    wts_add_np = wts_add.data.numpy()
    wts_add_list = wts_add_np.tolist()
    new_wts = []
    for s_id in range(len(maps)):
        new_wts.append([wts_add_list[s_id][i] for i in maps[s_id]])
    text= []
    for index, test in enumerate(words):
        text.append(" ".join(test))
    fileVizMap[viz_type]['text'].extend(text)
    fileVizMap[viz_type]['wts'].extend(new_wts)
    attention_visualization.createHTML(text, new_wts, filename)
    return


def to_tensor_labels(labels,  MAX_LEN):
    maxlen = MAX_LEN
    tensor =[]
    for i, sample in enumerate(labels):
        seq_len = len(sample)
        padding_len = abs(seq_len - maxlen)
        pad = [[1,0]] * padding_len
        sample.extend(pad)
        tensor.append(sample)
    tensor_tensor = torch.Tensor(tensor)

    if torch.cuda.is_available():
        tensor_tensor = tensor_tensor.cuda()
    return  tensor_tensor


def generate_bert_tensor_data(dataset, tokenizer, MAX_LEN):
    """Generators tensors for BERT
    """
    # initializes dataset.bert_tokens, dataset.bert_labels
    dataset.generate_bert_tokens(tokenizer)
    bert_token_ids = []
    attn_masks = []
    segment_ids = []
    for tokens in dataset.bert_tokens:
        padded_tokens = tokens + ['[PAD]' for _ in range(MAX_LEN - len(tokens))]
        attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
        seg_ids = [0 for _ in range(len(padded_tokens))]
        token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
        bert_token_ids.append(token_ids)
        attn_masks.append(attn_mask)
        segment_ids.append(seg_ids)
    bert_labels = to_tensor_labels(dataset.bert_labels, MAX_LEN)
    return torch.tensor(bert_token_ids), torch.tensor(segment_ids), torch.tensor(attn_masks), bert_labels


def check_predictions(preds, targets, mask):
    overlaped = (preds == targets)
    right = np.sum(overlaped * mask)
    total = mask.sum()
    return right, total, (overlaped * mask)

def get_batch_all_label_pred(numpy_predictions, numpy_label, mask_numpy, scores_numpy=None):
    """
    To remove paddings
    :param numpy_predictions:
    :param numpy_label:
    :param mask_numpy:
    :param scores_numpy: need this for computing ROC curve
    :return:
    """
    all_label =[]
    all_pred =[]
    all_score = []
    for i in range(len(mask_numpy)):

        all_label.append(list(numpy_label[i][:mask_numpy[i].sum()]))
        all_pred.append(list(numpy_predictions[i][:mask_numpy[i].sum()]))
        if isinstance(scores_numpy, np.ndarray):
            all_score.append(list(scores_numpy[i][:mask_numpy[i].sum()]))

        assert(len(list(numpy_label[i][:mask_numpy[i].sum()]))==len(list(numpy_predictions[i][:mask_numpy[i].sum()])))
        if isinstance(scores_numpy, np.ndarray):
            assert(len(list(numpy_label[i][:mask_numpy[i].sum()])) == len(list(scores_numpy[i][:mask_numpy[i].sum()])))
        assert(len(all_label)==len(all_pred))
    return  (all_label, all_pred) if not isinstance(scores_numpy, np.ndarray) else (all_label, all_pred, all_score)


def train(model, tr_dataloader, dev_dataloader,corpus, epochs=1, learning_rate=0.0001):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    # if FULL_FINETUNING:
    #     param_optimizer = list(model.named_parameters())
    #     no_decay = ['bias', 'gamma', 'beta']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.01},
    #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.0}
    #     ]
    # else:
    #     param_optimizer = list(model.classifier.named_parameters())
    #     optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.KLDivLoss(reduction='batchmean')

    # Training
    max_grad_norm = 1.0
    for epoch in trange(epochs, desc="Epoch"):
        # TRAIN loop
        train_total_preds = 0
        train_right_preds = 0
        total_train_loss =0
        train_total_y_true = []
        train_total_y_pred =[]
        model.train()
        for step, batch in enumerate(tqdm(tr_dataloader, desc="Batch")):
            optimizer.zero_grad()
            # add batch to gpu
            b_inputs, b_topic_ids, b_masks, b_labels = batch
            batch_size, seq_len = b_labels.size(0), b_labels.size(1)
            # forward pass
            scores = model.forward(b_inputs, b_masks, b_topic_ids)[0]
            # score_flat
            scores_flat = F.log_softmax(scores.view(batch_size * seq_len, -1), dim=1)
            # target_flat shape= [batch_size * seq_len]
            target_flat = b_labels.view(batch_size * seq_len, 2)
            train_loss = loss_func(scores_flat, F.softmax(target_flat,dim=1))
            target_flat_softmaxed = F.softmax(target_flat, 1)
            # backward pass
            train_loss.backward()
            optimizer.step()

            # track train loss
            total_train_loss += train_loss.item() * batch_size
            # get the index of the label with higher probability - used for AUC-ROC score.
            _, predictions_max = torch.max(torch.exp(scores_flat), 1)
            predictions_max = predictions_max.view(batch_size, seq_len)
            numpy_predictions_max = predictions_max.cpu().detach().numpy()

            _, label_max = torch.max(target_flat_softmaxed, 1)
            label_max = label_max.view(batch_size, seq_len)
            numpy_label_max = label_max.cpu().detach().numpy()

            mask_numpy = b_masks.cpu().detach().numpy()
            right, whole, overlaped = check_predictions(numpy_predictions_max, numpy_label_max, mask_numpy)
            train_total_preds += whole
            train_right_preds += right
            all_label, all_pred = get_batch_all_label_pred(numpy_predictions_max, numpy_label_max, mask_numpy)
            train_total_y_pred.extend(all_pred)
            train_total_y_true.extend(all_label)


        # print train loss per epoch
        train_f1_total = metrics.flat_f1_score(train_total_y_true, train_total_y_pred, average= "micro")

        train_loss = total_train_loss/ len(corpus.train.labels)
        print("[lOG] ++Train_loss: {}++, ++MAX train_accuracy: {}++, ++MAX train_f1_score: {}++ ".format(train_loss, (train_right_preds / train_total_preds), (train_f1_total) ))



        # VALIDATION on validation set
        print("[LOG] ______compute dev: ")
        model.eval()
        dev_right_preds = 0
        dev_total_preds = 0
        total_dev_loss = 0
        dev_total_y_true = []
        dev_total_y_pred = []

        for batch in dev_dataloader:
            b_inputs, b_topic_ids, b_masks, b_labels = batch
            batch_size, seq_len = b_labels.size(0), b_labels.size(1)
            dev_score = model.forward(b_inputs, b_masks, b_topic_ids)[0]
            # score_flat
            dev_scores_flat = F.log_softmax(dev_score.view(batch_size * seq_len, -1), dim=1)
            # target_flat shape= [batch_size * seq_len]
            dev_target_flat = b_labels.view(batch_size * seq_len, 2)
            dev_loss = loss_func(dev_scores_flat, F.softmax(dev_target_flat,dim=1))
            dev_target_flat_softmaxed = F.softmax(dev_target_flat, 1)
            total_dev_loss += dev_loss.item() * batch_size
            dev_target_flat_softmaxed = F.softmax(dev_target_flat, 1)

            _, dev_predictions_max = torch.max(dev_scores_flat, 1)
            dev_predictions_max = dev_predictions_max.view(batch_size, seq_len)
            dev_numpy_predictions_max = dev_predictions_max.cpu().detach().numpy()


            _, dev_label_max = torch.max(dev_target_flat_softmaxed, 1)
            dev_label_max = dev_label_max.view(batch_size, seq_len)
            dev_numpy_label_max = dev_label_max.cpu().detach().numpy()


            # mask:
            dev_mask_numpy = b_masks.cpu().detach().numpy()

            dev_right, dev_whole, dev_overlaped = check_predictions(dev_numpy_predictions_max, dev_numpy_label_max, dev_mask_numpy)
            dev_total_preds += dev_whole
            dev_right_preds += dev_right

            all_label, all_pred = get_batch_all_label_pred(dev_numpy_predictions_max, dev_numpy_label_max, dev_mask_numpy, 0)
            dev_total_y_pred.extend(all_pred)
            dev_total_y_true.extend(all_label)
        else:
            dev_f1_total_micro = metrics.flat_f1_score(dev_total_y_true, dev_total_y_pred, average= "micro")

        dev_loss = total_dev_loss / len(corpus.dev.labels)
        dev_f1_total_macro = metrics.flat_f1_score(dev_total_y_true, dev_total_y_pred, average="macro")
        #checkpoint:
        is_best = helper.checkpoint_model(model, optimizer, config.output_dir_path, dev_loss, epoch + 1, 'min')

        print("<<dev_loss: {}>> <<dev_accuracy: {}>> <<dev_f1: {}>> ".format( dev_loss, (dev_right_preds / dev_total_preds), (dev_f1_total_micro)))
        print("--------------------------------------------------------------------------------------------------------------------------------------------------")
        #tensorBoard:
        info = {'training_loss': train_loss,
                'train_accuracy': (train_right_preds / train_total_preds),
                'train_f1': (train_f1_total),
                'validation_loss': dev_loss,
                'validation_accuracy': (dev_right_preds / dev_total_preds),
                'validation_f1_micro': (dev_f1_total_micro),
                'validation_f1_macro': (dev_f1_total_macro)
                }
        tensor_logging(model, info, epoch)
    print(f"Best model saved to: {config.output_dir_path + 'best.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_viz", action="store_true",
                        help="Whether to run visualization for test set.")
    parser.add_argument("--dump_viz", action="store_true",
                        help="Whether store the results of visualizations.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Override batch_size")
    parser.add_argument("--learning_rate", default=0.0001, type=float,
                        help="Override learning ratene")
    parser.add_argument("--epochs", default=3, type=int,
                        help="Override number of epochs")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to run training.")
    args = parser.parse_args()

    lr = args.learning_rate
    epochs = args.epochs
    bs = args.batch_size
    corpus = Corpus.get_corpus(config.corpus_dir, config.corpus_pkl)

    # Load bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_directory)


    tr_inputs, tr_topic_ids, tr_masks, tr_labels  = generate_bert_tensor_data(corpus.train, tokenizer, MAX_LEN)

    dev_inputs, dev_topic_ids, dev_masks, dev_labels  = generate_bert_tensor_data(corpus.dev, tokenizer, MAX_LEN)
    test_inputs, test_topic_ids, test_masks, test_labels  = generate_bert_tensor_data(corpus.test, tokenizer, MAX_LEN)
    tr_data = TensorDataset(tr_inputs, tr_topic_ids, tr_masks, tr_labels)
    tr_sampler = RandomSampler(tr_data)
    tr_dataloader = DataLoader(tr_data, sampler=tr_sampler, batch_size=bs)

    dev_data = TensorDataset(dev_inputs, dev_topic_ids, dev_masks, dev_labels)
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=bs)

    test_data = TensorDataset(test_inputs, test_topic_ids, test_masks, test_labels)
    # test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=bs)

    # model = BertForTokenClassification.from_pretrained(bert_directory, num_labels=NUM_LABELS)

    model = BertAttnModel(len(corpus.get_label_vocab()), config.extractor_type,  config.hidden_dim)

    if args.do_train:
        print('Running training....')
        train(model, tr_dataloader, dev_dataloader, corpus, epochs=epochs, learning_rate=lr)

    # Predict on Test data
    if not args.do_test:
        import sys
        sys.exit(0)
    print('Evaluating test data....')
    loss_func = nn.KLDivLoss(reduction='batchmean')
    helper.load_saved_model(model, config.output_dir_path + 'best.pth')
    model.eval()
    total_batch_test = len(corpus.test.labels) // bs
    if len(corpus.test.words) % bs > 0:
        total_batch_test += 1

    test_right_preds, test_total_preds = 0, 0
    test_total_y_true = []
    test_total_y_pred = []
    test_total_y_scores = []
    total_scores_numpy_probs =[]
    total_labels_numpy_probs =[]
    total_mask_numpy =[]
    total_test_loss = 0
    batch_i = -1
    for batch in tqdm(test_dataloader, desc="batch"):
        batch_i += 1
        b_inputs, b_topic_ids, b_masks, b_labels = batch
        batch_size, seq_len = b_labels.size(0), b_labels.size(1)
        batch_start = batch_i * batch_size
        batch_end = batch_start + batch_size
        b_words = corpus.test.words[batch_start: batch_end]
        b_id_maps = corpus.test.bert_token_maps[batch_start: batch_end]
        # forward pass
        scores = model(b_inputs, b_masks, b_topic_ids)[0]
        # score_flat
        scores_flat = F.log_softmax(scores.view(batch_size * seq_len, -1), dim=1)
        # target_flat shape= [batch_size * seq_len]
        target_flat = b_labels.view(batch_size * seq_len, 2)

        test_loss = loss_func(scores_flat, F.softmax(target_flat,dim=1))
        target_flat_softmaxed = F.softmax(target_flat, 1)
        total_test_loss += test_loss.item() * batch_size
        scores_flat_exp = torch.exp(scores_flat)

        print("--[LOG]-- test loss: ", test_loss)

        _, predictions_max = torch.max(scores_flat_exp, 1)

        predictions_max = predictions_max.view(batch_size, seq_len)
        numpy_predictions_max = predictions_max.cpu().detach().numpy()

        # Visualization:
        if args.do_viz:
            sfe = scores_flat_exp[:, 1].view(batch_size, seq_len)

            visualize_attention(sfe, b_words, filename='res/' + config.testing + '_scores'+str(batch_i)+'.html', maps=b_id_maps, viz_type='scores')
            visualize_attention(b_labels[:,:,1], b_words, filename='res/target' + str(batch_i) + '.html', maps=b_id_maps, viz_type='target')
            visualize_attention(F.softmax(b_labels, 1)[:,:,1], b_words, filename='res/target_softmaxed' + str(batch_i) + '.html', maps=b_id_maps, viz_type='target_softmax')
        # computing scores for ROC curve:
        scores_numpy = scores_flat_exp[:, 1].view(batch_size, seq_len)
        scores_numpy = scores_numpy.cpu().detach().numpy()

        total_scores_numpy_probs.extend(scores_numpy)

        # if based on MAX
        _, label_max = torch.max(target_flat, 1)
        label_max = label_max.view(batch_size, seq_len)
        numpy_label_max = label_max.cpu().detach().numpy()
        # for computing senetnce leveL:
        total_labels_numpy_probs.extend(target_flat[:, 1].view(batch_size, seq_len).cpu().detach().numpy())

        # mask:
        mask_numpy = b_masks.cpu().detach().numpy()
        total_mask_numpy.extend(mask_numpy)
        right, whole, overlaped = check_predictions(numpy_predictions_max, numpy_label_max, mask_numpy)
        test_total_preds += whole
        test_right_preds += right
        all_label, all_pred, all_scores= get_batch_all_label_pred(numpy_predictions_max, numpy_label_max, mask_numpy, scores_numpy)
        test_total_y_pred.extend(all_pred)
        test_total_y_true.extend(all_label)
        test_total_y_scores.extend(all_scores)

    test_f1_total_micro = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average= "micro")
    test_f1_total_macro = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average="macro")
    test_f1_total_binary = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average="binary")

    roc_score= roc_auc_score(list(itertools.chain(*test_total_y_true)) , list(itertools.chain(*test_total_y_scores)))
    test_loss = total_test_loss / len(corpus.test.labels)

    print(
        "->>>>>>>>>>>>>TOTAL>>>>>>>>>>>>>>>>>>>>>>> test_loss: {}, test_accuracy: {}, test_f1_score_micro: {} ROC:{}".format(
            test_loss, (test_right_preds / test_total_preds), (test_f1_total_micro), roc_score))
    print()
    print(metrics.flat_classification_report(test_total_y_true, test_total_y_pred))
    print("test_f1_total_binary: ", test_f1_total_binary)
    print("precision binary: ", metrics.flat_precision_score(test_total_y_true, test_total_y_pred, average="binary"))
    print("recall binary: ", metrics.flat_recall_score(test_total_y_true, test_total_y_pred, average="binary"))


    if not os.path.exists(config.dump_address):
        os.makedirs(config.dump_address)
    print("[LOG] dumping results in ", config.dump_address)
    pickle.dump(np.array(total_scores_numpy_probs),
                open(os.path.join(config.dump_address, "score_pobs.pkl"), "wb"))
    pickle.dump(np.array(total_labels_numpy_probs),
                open(os.path.join(config.dump_address, "label_pobs.pkl"), "wb"))
    pickle.dump(np.array(total_mask_numpy), open(os.path.join(config.dump_address, "mask_pobs.pkl"), "wb"))

    if args.do_viz and args.dump_viz:
        pickle.dump(fileVizMap, open(os.path.join(config.dump_address, "vizmap.pkl"), "wb"))
        attention_visualization.createHTML(fileVizMap['scores']['text'], fileVizMap['scores']['wts'], 'res/' + config.testing + '_scores_all.html')
        attention_visualization.createHTML(fileVizMap['target']['text'], fileVizMap['target']['wts'], 'res/TargetAll.html')
        attention_visualization.createHTML(fileVizMap['target_softmax']['text'], fileVizMap['target_softmax']['wts'], 'res/TargetSoftmaxAll.html')

        import csv

        with open('./visualization/res/targetAll.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            for row in fileVizMap['target']['wts']:
                writer.writerow(row)

        with open('./visualization/res/' + config.testing + '_All.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            for row in fileVizMap['scores']['wts']:
                writer.writerow(row)

        with open('./visualization/res/targetSoftAll.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=',')
            for row in fileVizMap['target_softmax']['wts']:
                writer.writerow(row)

        with open('./visualization/res/words.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=' ')
            for row in fileVizMap['target']['text']:
                writer.writerow(row.split(' '))

    embed()
