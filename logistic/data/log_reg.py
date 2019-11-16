from collections import defaultdict
from collections import OrderedDict
import spacy
print("loading spacy")
nlp = spacy.load('en_core_web_md')
print("finished loading")
ip_file = "bio_probs.txt"
pos_tags = []   
X_train, y_train = [], []
keylist = []
word_dict = defaultdict(list)

def word_pos(tag):
    pos_emb = []
    for pos in pos_tags:
        if tag == pos:
            pos_emb.append(1) 
        else:
            pos_emb.append(0) 
    return pos_emb

def word_emb(word):
    emb = []
    return emb

with open(ip_file, 'r') as fp:
    lines = [line.split() for line in fp]


for i in lines:
    if i:
        pos_tags.append(i[4])
        
pos_tags = list(set(pos_tags))

outer_counter = 0
for l in lines:
    if not l:
        inner_counter = 1
        outer_counter+=1
    else:
        w_id = str(outer_counter)+"_"+str(inner_counter)
        word = l[0]
        pos = l[4]
        y = l[3]
        keylist.append(w_id)
        word_dict[w_id].append(word)
        word_dict[w_id].append(pos)
        word_dict[w_id].append(y)
        inner_counter+=1
        
        
for index,key in enumerate(word_dict):
    print(str(index+1)+" values seen")
    outer, inner = key.split("_")
    #prev pos embedding
    if inner=="1":
        prev_pos_emb = [0] * len(pos_tags)
        
    else:
        prev_pos = word_dict[keylist[index-1]][1]
        prev_pos_emb = word_pos(prev_pos)
        

    #current pos embedding
    cur_pos = word_dict[key][1]
    cur_pos_emb = word_pos(cur_pos)
    

    #next pos embedding
    next_outer = str(int(outer)+1)+"_1"
    
    if next_outer == keylist[index+1]:
        next_pos_emb = [0] * len(pos_tags)
    else:
        next_pos = word_dict[keylist[index+1]][1]
        next_pos_emb = word_pos(next_pos)

    # get word-embedding for particular word
    cur_word = word_emb(word_dict[key][0])   
    word_vector = list(cur_word.vector)
    print("appending into train")
    #X_train
    X_train.append(word_vector+prev_pos_emb+cur_pos_emb+next_pos_emb)
    print(X_train)
    #y-values
    true_y = 1 if float(word_dict[key][2]) >= 0.5 else 0
    
    #y-train
    y_train.append(true_y)


print(X_train[0])
