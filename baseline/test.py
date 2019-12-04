from utils.data import *
from IPython import embed

corpus_dir = '../Data/formatted'
emb_path = '../../embedding/glove.6B.100d.txt'
corpus_pkl = corpus_dir + '/corpus_pkl.pkl'

# Read in the corpus
corpus = Corpus.get_corpus(corpus_dir, corpus_pkl)
embed()