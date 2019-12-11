import bert.config
from bert.model import BertAttnModel
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import bert.helper as helper
import torch
import torch.nn.functional as F
from bert.main import *
import numpy as np

MAX_LEN = 75

def normalize(v):
    den = np.max(v) - np.min(v)
    return list((v-np.min(v)) / den)

class BertPredictor:
    def __init__(self):
        self.model = BertAttnModel(2, config.extractor_type,  config.hidden_dim)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_directory)
        self._model_loaded = False

    def load_model(self, model_path):
        """Loads the trained BERT model from model_path"""
        self.model_path = model_path
        try:
            helper.load_saved_model(self.model, model_path)
            self.model.eval()
            self._model_loaded = True

        except Exception as e:
            print(e)
            self._model_loaded = False
            return False

    def _bert_tokenize_sentence(self, sentence):
        """Returns bert tokens and labels split using bert tokenizer,
           with map between original and new labels
           for one sentence
           Example: "Welcome to housewarming party"
            ['CLS', 'Welcome', 'to', 'house#####', '#####warming', 'party', [SEP]]
            [1, 2, 4,  5]


        Arguments:
            tokenizer, BertTokenizer: the initialized BertTokenizer object.
            sentence, List[str]: list of original tokens
        Returns:
            bert_tokens, List[str]: The list of bert tokens
            index_map, List[int]: Maps original word indices in sentence to new bert_token indices
        """
        ### Output
        bert_tokens = []

        # Token map will be an int -> int mapping between the `orig_tokens` index and
        # the `bert_tokens` index.
        index_map = []
        bert_tokens.append("[CLS]")
        for word in sentence:
            index_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer.tokenize(word))
        bert_tokens.append("[SEP]")
        return bert_tokens, index_map

    def _bert_tokenize(self, sentence):
        """Generates Bert tokens"""
        batch_tokens = []
        id_maps = []
        s_tokens, id_map = self._bert_tokenize_sentence(sentence)
        batch_tokens.append(s_tokens)
        id_maps.append(id_map)
        bert_token_ids = []
        attn_masks = []
        segment_ids = []
        for tokens in batch_tokens:
            padded_tokens = tokens + ['[PAD]' for _ in range(MAX_LEN - len(tokens))]
            attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
            seg_ids = [0 for _ in range(len(padded_tokens))]
            token_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
            bert_token_ids.append(token_ids)
            attn_masks.append(attn_mask)
            segment_ids.append(seg_ids)

        return torch.tensor(bert_token_ids), torch.tensor(segment_ids), torch.tensor(attn_masks), id_maps

    def predict(self, text):
        """Returns emphasis probablility for each word in text using loaded BERT model"""
        if self._model_loaded:
            sentence = text.split(' ')
            b_inputs, b_topic_ids, b_masks, id_maps = self._bert_tokenize(sentence)
            scores = self.model(b_inputs, b_masks, b_topic_ids)[0]
            scores_flat = F.log_softmax(scores.view(1 * MAX_LEN, -1), dim=1)
            scores_flat_exp = torch.exp(scores_flat)
            wts = scores_flat_exp[:, 1].view(1, MAX_LEN)
            wts_add = wts.cpu()
            wts_add_np = wts_add.data.numpy()
            wts_add_list = wts_add_np.tolist()
            new_wts = np.array([wts_add_list[0][i] for i in id_maps[0]])
            # new_wts = normalize(new_wts)
            return list(new_wts)
        return []

