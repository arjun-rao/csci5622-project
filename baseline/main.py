import os
import argparse
import torch
import json
import re
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.optim import SGD
from utils.data  import Corpus, Encoder
from model.seqmodel import SeqModel
from model.seqmodel_Elmo import SeqModel_Elmo
from model.seqmodel_Bert import SeqModel_Bert
#from model.lstm_crf import Lstm_crf
import torch.optim as optim
import config
from config import *
from IPython import embed

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Running on GPU {}".format(gpu_number))
        torch.cuda.set_device(gpu_number)
    else:
        print("Running on CPU")

    print("[LOG] dumping in .. ", config.dump_address)
    if not config.training:
        print("[LOG] NO training ...!")



    torch.manual_seed(0)
    np.random.seed(0)



    corpus = Corpus.get_corpus(corpus_dir, corpus_pkl)

    if config.if_flair:
        # encoder = Encoder(corpus, emb_path, flair=True)
        # with open(corpus_pkl_flair, 'wb') as fp:
        #     pickle.dump(corpus, fp, -1)

        with open(corpus_pkl_flair, 'rb') as fp:
            corpus = pickle.load(fp)
        encoder = None
    else:
        encoder = Encoder.get_encoder(corpus, emb_path, encoder_pkl)

        if not (config.if_Elmo or config.if_Bert):
            encoder.encode_words(corpus, flair=True)

        embed()

    if model_mode=="prob":
        from trainer_prob import Trainer
        theLoss = nn.KLDivLoss(reduction='elementwise_mean')#size_average=True)
        if config.if_Elmo:
            print("[LOG] Using Elmo ...")
            model = SeqModel_Elmo(len(corpus.get_label_vocab()), extractor_type,  hidden_dim)
        elif config.if_Bert:
            print("[LOG] Using Bert ...")
            model = SeqModel_Bert(len(corpus.get_label_vocab()), extractor_type,  hidden_dim)
        else:
            if config.if_flair:
                print("[LOG] Using Flair ...")
            model = SeqModel(len(corpus.get_label_vocab()), extractor_type,  hidden_dim)
        optimizer = optim.Adam(lr=lr, params=model.parameters())
        print("==========================================================")
        print("[LOG] Model:")
        print(model)
        print("==========================================================")
        print("[LOG] Train:")
        trainer = Trainer(corpus, encoder, batch_size, epochs)
        if config.training:
            trainer.train(model, theLoss, optimizer)


        print("==========================================================")
        print("[LOG] Test:")
        trainer.predict(model, theLoss, trainer.corpus.test)
        embed()



