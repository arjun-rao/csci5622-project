import os
import pickle

from collections import Counter


def flatten(elems):
    return [e for elem in elems for e in elem]

class Dataset(object):
    def __init__(self, path):
        self.words  = self.read_conll_format(os.path.join(path, 'bio_probs.txt'))
        self.labels = self.read_conll_format_labels(os.path.join(path, 'bio_probs.txt'))
        self.tags = self.labels2tags(self.labels)
        self.bert_labels = []
        self.bert_tokens = []
        self.bert_label_counts = []
        self.bert_token_maps = []
        assert len(self.words) == len(self.labels)
        assert len(self.words) == len(self.tags)

    def labels2tags(self, label_probs):
        tags = []
        for sentence in label_probs:
            sentence_tags = []
            for label in sentence:
                if label[0] > 0.5:
                    sentence_tags.append('O')
                else:
                    sentence_tags.append('I')
            tags.append(sentence_tags)
        return tags

    def generate_bert_tokens(self, tokenizer):
        self.bert_tokens, self.bert_labels, self.bert_token_maps = self._bert_tokenize(tokenizer)

    def _bert_tokenize_sentence(self, tokenizer, sentence):
        """Returns bert tokens and labels split using bert tokenizer, with map between original and new labels
        for one sentence

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
            bert_tokens.extend(tokenizer.tokenize(word))
        bert_tokens.append("[SEP]")
        return bert_tokens, index_map

    def _bert_tokenize(self, tokenizer):
        tokens = []
        labels = []
        id_maps = []
        for s_idx, sentence in enumerate(self.words):
            s_tokens, id_map = self._bert_tokenize_sentence(tokenizer, sentence)
            orig_labels = self.labels[s_idx]
            s_labels = [[1, 0]] * len(s_tokens)
            for i in range(len(id_map)):
                s_labels[id_map[i]] = orig_labels[i]
            tokens.append(s_tokens)
            labels.append(s_labels)
            id_maps.append(id_map)
        return tokens, labels, id_maps


    def get_bert_tokens(self, tokenizer):
        labels = []
        tokens = []
        label_counts = []
        for s_idx, sentence in enumerate(self.words):
            s_labels = []
            l_count = []
            for w_idx, word in enumerate(sentence):
                label = self.tags[s_idx][w_idx]
                count = len(tokenizer.tokenize(word))
                l_count.append(count)
                s_labels.extend([label] * count)
            bert_tokens = tokenizer.tokenize(' '.join(sentence))
            tokens.append(bert_tokens)
            labels.append(s_labels)
            label_counts.append(l_count)
            assert len(bert_tokens) == len(s_labels)
        return labels, tokens, label_counts

    def read_conll_format_labels(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                probs = line.split("\t")[2]
                # reading probabilities from the last column and also normalaize it by div on 9
                probs = [(int(l)/9) for l in probs.split("|")]
                probs = [probs[2],probs[0]+probs[1] ]

                post.append(probs)
                print("post: ", post)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_conll_format(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                words = line.split("\t")[0]
                # print("words: ", words)
                post.append(words)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines

class Corpus(object):
    def __init__(self, corpus_path):
        self.train = Dataset(os.path.join(corpus_path, 'train/'))
        self.dev = Dataset(os.path.join(corpus_path, 'dev/'))
        self.test = Dataset(os.path.join(corpus_path, 'test/'))

    @staticmethod
    def get_corpus(corpus_dir, corpus_pkl_path):
        if os.path.exists(corpus_pkl_path):
            with open(corpus_pkl_path, 'rb') as fp:
                corpus= pickle.load(fp)

        else:
            corpus = Corpus(corpus_dir)
            with open(corpus_pkl_path, 'wb') as fp:
                pickle.dump(corpus, fp, -1)
        corpus.print_stats()
        return corpus

    @staticmethod
    def _get_unique(elems):
        corpus = flatten(elems)
        elems, freqs = zip(*Counter(corpus).most_common())
        return list(elems)


    def print_stats(self):

        print("Train dataset: {}".format(len(self.train.words)))
        print("Dev dataset: {}".format(len(self.dev.words)))
        print("Test dataset: {}".format(len(self.test.words)))

    def get_word_vocab(self):
        return self._get_unique(self.train.words + self.dev.words + self.test.words)
    def get_label_vocab(self):
        return self._get_unique(["O", "I"])

