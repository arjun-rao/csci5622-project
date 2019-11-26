from collections import defaultdict
from collections import OrderedDict
from IPython import embed
from tqdm import tqdm
# import spacy
# nlp = spacy.load('en_core_web_md')
from data import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

class Pos:
    def __init__(self):
        self.word = 0
        self.label = 3
        self.pos = 4
        
class Logistic:
    def __init__(self):        
        self.emb_path = '../../../embedding/glove.6B.100d.txt'
        self.corpus_dir = '../../Data/formatted/'
        self.corpus_pkl =  "./corpus.io.pkl"
        self.encoder_pkl = "./encoder.io.pkl"
        self.corpus = Corpus.get_corpus(self.corpus_dir, self.corpus_pkl)
        self.encoder = Encoder.get_encoder(self.corpus, self.emb_path, self.encoder_pkl)
        
    def word_pos(self,tag, pos_tags):
        pos_emb = []
        for pos in pos_tags:
            if tag == pos:
                pos_emb.append(1)
            else:
                pos_emb.append(0)
        return pos_emb

    def word_emb(self,word):
        emb = self.encoder.word_emb[self.encoder.word2index[word]]
        return emb

    def generate_features(self,filename, pos_tags = None,prev_count=1,next_count=1):
        pos_obj = Pos()
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
                word = sentence[pos_obj.word]
                pos = sentence[pos_obj.pos]
                y = sentence[pos_obj.label]
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
            if inner=="1":  #handles first word with no-prev embedding
                prev_pos_emb = [0] * len(pos_tags) * prev_count

            else:
                prev_pos = word_dict[keylist[index-1]][1]
                prev_pos_emb = self.word_pos(prev_pos, pos_tags)


            #current pos embedding
            cur_pos = word_dict[key][1]
            cur_pos_emb = self.word_pos(cur_pos, pos_tags)

            #next pos embedding
            next_outer = str(int(outer)+1)+"_1"
            try:
                if next_outer == keylist[index+1]:
                    next_pos_emb = [0] * len(pos_tags) * next_count
                else:
                    next_pos = word_dict[keylist[index+1]][1]
                    next_pos_emb = self.word_pos(next_pos, pos_tags)
            except:
                next_pos_emb = [0] * len(pos_tags) * next_count

            # get word-embedding for particular word
            word_vector = list(self.word_emb(word_dict[key][0]))

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
    
    def predict_score(self,predicted_scores,true_label):
        match_m_score = match_M(predicted_scores,true_label)
        top_k_score = topK(predicted_scores,true_label)
        return match_m_score, top_k_score


def main():
    lr_obj = Logistic()
    x_train, y_train, xy_train, pos_tags = lr_obj.generate_features('./bio_probs_train.txt')

    x_test, y_test, xy_test, _ = lr_obj.generate_features('./bio_probs_test.txt', pos_tags)
    clf = LogisticRegression(random_state=2019)
    clf.fit(x_train, y_train)
    y_pred = []
    y_test_prob = []
    #stores sentence features, sentence_probs, word_dict respectively
    x_test, y_test, word_dict = xy_test
    
    for i in x_test:
        labels = clf.predict_proba(x_test[i])
        y_pred.append([item[1] for item in labels])
        y_test_prob.append(y_test[i])
    
    match_score,k_score = lr_obj.predict_score(y_pred,y_test_prob)
    print("Match Score: ",match_score)
    print("K score: ",k_score)
    # embed()

if __name__ == "__main__":
    main()