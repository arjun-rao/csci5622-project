from collections import defaultdict
from collections import OrderedDict
from IPython import embed
from tqdm import tqdm
# import spacy
# nlp = spacy.load('en_core_web_md')
from data import *
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import fasttext
from visualization import attention_visualization
import torch

emb_path = '../../../embedding/glove.6B.100d.txt'
corpus_dir = '../../Data/formatted/'
corpus_pkl =  "./corpus.io.pkl"
encoder_pkl = "./encoder.io.pkl"

corpus = Corpus.get_corpus(corpus_dir, corpus_pkl)
encoder = Encoder.get_encoder(corpus, emb_path, encoder_pkl)
model = fasttext.load_model("result/fil9.bin")




def word_emb(word,glove_emb=True):
        if glove_emb:
            emb = encoder.word_emb[encoder.word2index[word]]
        else:
            emb = model.get_word_vector(word)
        return emb

def word_pos(tag, pos_tags):
    pos_emb = []
    for pos in pos_tags:
        if tag == pos:
            pos_emb.append(1)
        else:
            pos_emb.append(0)
    return pos_emb



def generate_features(filename, pos_tags = None,prev_count=1,next_count=1,glove_emb=True):
    ip_file = filename
    X_split, y_split = [], []
    keylist = []
    word_dict = defaultdict(list)   #key:sentence_word_id; value:["word",pos_tag,predicted_label]
    
    with open(ip_file, 'r') as fp:
        lines = [line.split() for line in fp]
    
    #generate list of unique pos-tags
    if pos_tags is None:
        pos_tags = []
        for i in lines:
            if i:
                pos_tags.append(i[4])
    
        pos_tags = list(set(pos_tags))
    
    #generate feature dict
    #outer-counter handles lines, inner-counter handles each word in line
    outer_counter = 0
    for sentence in lines:
        if not sentence: #handles blank lines; separator between different sentences
            inner_counter = 1
            outer_counter+=1
        else:
            word_id = str(outer_counter)+"_"+str(inner_counter)
            word = sentence[0]
            pos = sentence[4]
            y = sentence[3]
            keylist.append(word_id)
            word_dict[word_id].append(word)
            word_dict[word_id].append(pos)
            word_dict[word_id].append(y)
            inner_counter+=1
    key_indices = list(enumerate(word_dict))
    # current_key = key_indices[0][1]
    sentence_features = defaultdict(list)
    sentence_probs = defaultdict(list)
    for index,key in tqdm(key_indices):
        outer, inner = key.split("_")
        #prev pos embedding
        # prev_pos_emb = [0] * len(pos_tags) * prev_count
        if inner=="1":  #handles first word with no-prev embedding
            prev_pos_emb = [0] * len(pos_tags) * prev_count
    
        else:
            prev_pos_emb = []
            for i in range(prev_count+1):
                if i!=0 and index-i >=0:
                    prev_pos = word_dict[keylist[index-i]][1]
                    prev_pos_emb.extend(word_pos(prev_pos, pos_tags))
                elif index-i<0:
                    prev_pos_emb.extend([0]*len(pos_tags))
    
    
        #current pos embedding
        cur_pos = word_dict[key][1]
        cur_pos_emb = word_pos(cur_pos, pos_tags)
    
    
        #next pos embedding
        next_outer = str(int(outer)+1)+"_1"
        try:
            next_pos_emb=[]
            for i in range(next_count+1):
                if next_outer == keylist[index+i] or next_outer.split("_")[0] == keylist[index+i].split("_")[0]:
                    next_pos_emb = [0] * len(pos_tags) * next_count
                elif i!=0:
                    # next_pos_emb = []
                    next_pos = word_dict[keylist[index+i]][1]
                    next_pos_emb.extend(word_pos(next_pos, pos_tags))
        except:
            next_pos_emb = [0] * len(pos_tags) * next_count
    
        # get word-embedding for particular word
        list(word_emb(word_dict[key][0],glove_emb))
        word_vector = list(word_emb(word_dict[key][0],glove_emb))
        f_prev,f_cur,f_next = len(prev_pos_emb), len(cur_pos_emb),len(next_pos_emb)
        #X_train
        feature_vector = word_vector+prev_pos_emb+cur_pos_emb+next_pos_emb
        
        X_split.append(feature_vector)
        sentence_features[outer].append(feature_vector)
        
        #y-values
        true_y = 1 if float(word_dict[key][2]) >= 0.5 else 0
        sentence_probs[outer].append(float(word_dict[key][2]))
    
        #y-train
        y_split.append(true_y)
    return X_split, y_split, [sentence_features, sentence_probs, word_dict], pos_tags


class Xgboostclassifier:
    def __init__(self, n_clusters=8, k='deprecated'):
        self.n_clusters = n_clusters
        self.k = k

    def fit(self, X, y):
        if self.k != 'deprecated':
            warnings.warn("'k' was renamed to n_clusters in version 0.13 and "
                          "will be removed in 0.15.",
                          FutureWarning)
            self._n_clusters = self.k
        else:
            self._n_clusters = self.n_clusters

x_train, y_train, xy_train, pos_tags = generate_features(filename='./bio_probs_train.txt',glove_emb=True)
x_test, y_test, xy_test, _ =  generate_features(filename='./bio_probs_test.txt', pos_tags=pos_tags, glove_emb=True)

#f_train = open("train.txt","a+")

#for i in range(len(y_train)):
#	f_train.write("%d " % (y_train[i]))
#	for j in range(len(x_train[i])):
#			f_train.write("%d:%d " % (j,x_train[i][j]))
#	f_train.write("\n")

#f_test = open("test.txt","a+")

#for i in range(len(y_test)):
#	f_test.write("%d " % (y_test[i]))
#	for j in range(len(x_test[i])):
#			f_test.write("%d:%d " % (j,x_test[i][j]))
#	f_test.write("\n")



#for i in range(len(x_train[0])):
#	for j in range(len(x_train[i])):
#		f_features.write("%d " % (x_train[i][j]))
#	f_features.write("\n")

#for i in range(len(y_train)):
  #   f.write("%d" % (y_train[i]))
	# f.write("%d" % (x_train[i]))


#dtrain = xgb.DMatrix('data.txt')

x_train_np = np.asarray(x_train)
y_train_np = np.asarray(y_train)

num_round = 100
param = {'max_depth': 3, 'eta': 0.3, 'nthread': 4}
dtrain = xgb.DMatrix(x_train_np, label=y_train_np)
bst = xgb.train(param, dtrain, num_round)

#clf = XGBClassifier(random_state=2019)
#clf.fit(x_train, y_train)
y_pred = []
y_test_prob = []
x_test, y_test, word_dict = xy_test

x_train_np = np.asarray(x_test)
y_train_np = np.asarray(y_test)

#x_test_np = np.array(list(x_test.items()), dtype=dtype)
#y_test_np = np.array(list(y_test.items()), dtype=dtype)

#x_test_list = []
#y_test_list = []

#for k,v in x_test.items():
#    x_test_list.append(v)

#for k,v in y_test.items():
 #   y_test_list.append(v)

#x_test_np = np.asarray(x_test_list)
#y_test_np = np.asarray(y_test_list)

y_pred_roc = []




for i in x_test:
    np_sentence = np.asarray(x_test[i])
    np_labels = np.asarray(y_test[i])
    dtest = xgb.DMatrix(np_sentence)
    labels = bst.predict(dtest)
    #print(labels)
    y_pred_roc.extend(labels)
    #print(labels)
    y_pred.append([item for item in labels])
    y_test_prob.append(y_test[i])
    #print(y_test[i])
#y_pred_tensor = torch.FloatTensor()
#visualize_attention(y_pred_tensor,,)
text_flat = []
scores_flat = []

for test in corpus.test.words:
    text_flat.append(" ".join(test))
embed()
attention_visualization.createHTML(text_flat, y_pred, "res/xgboost.html")

y_pred_roc = np.array([1 if i >= 0.5 else 0 for i in y_pred_roc])
y_test_roc = []
for i in y_test_prob:
    y_test_roc.extend(i)
y_test_roc = np.array([1 if i >= 0.5 else 0 for i in y_test_roc])
match_M(y_pred,y_test_prob)
topK(y_pred,y_test_prob)
print(roc_auc_score(y_test_roc, y_pred_roc))

embed()