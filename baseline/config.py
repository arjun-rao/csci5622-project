import torch

gpu_number = 0

##########################################################
model_mode = "prob"
############################################################
testing = "Flair"
corpus_dir = '../Data/formatted/'
output_dir_path = "../models_checkpoints/"+ testing+"/"
dump_address = "../evals/"+testing+"/"


training = True

if_Elmo = False

if_Bert = False

if_att = True

if_flair = True

if_ROC = True

if_visualize = True

##############################################################
if model_mode== "prob":
    corpus_pkl = corpus_dir + "corpus.io.pkl"
    corpus_pkl_flair = corpus_dir + "corpus.flair.pkl"
    encoder_pkl = corpus_dir + "encoder.io.pkl"
##############################################################
lr = 0.0001
extractor_type = 'lstm'
feat_extractor = 'lstm'

if if_Elmo:
    hidden_dim = 2048
elif if_Bert:
    hidden_dim = 768
elif if_flair:
    hidden_dim = 4096
else:
    hidden_dim = 512

epochs = 2
batch_size = 16

######################################Elmo files##################################################
options_file = "../../embedding/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../../embedding/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
########################################################################################
if not torch.cuda.is_available():
    print("[LOG] running on CPU")
    emb_path = '../../embedding/glove.6B.100d.txt'
else:
    print("[LOG] running on GPU")
    emb_path = '../../embedding/glove.6B.100d.txt'


bert_directory = '../../embedding/bert/'
