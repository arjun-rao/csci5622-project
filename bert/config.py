
bert_directory = '../../embedding/bert_cased/'
corpus_dir = '../Data/formatted/'
corpus_pkl = corpus_dir + "bert_corpus.io.pkl"
testing = "BertSolo"
output_dir_path = "../models_checkpoints/"+ testing+"/"
dump_address = "../evals/"+testing+"/"

FULL_FINETUNING = True

if_att = False
if_bilstm = False

extractor_type = 'lstm'
feat_extractor = 'lstm'
hidden_dim = 768
