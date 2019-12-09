
bert_directory = '../../embedding/bert_cased/'
corpus_dir = '../Data/formatted/'
corpus_pkl = corpus_dir + "bert_corpus.io.pkl"
testing = "BertBest"
output_dir_path = "../models_checkpoints/"+ testing+"/"
dump_address = "../evals/"+testing+"/"

FULL_FINETUNING = False

if_att = True

extractor_type = 'lstm'
feat_extractor = 'lstm'
hidden_dim = 768
