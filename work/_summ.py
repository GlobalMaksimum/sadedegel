# pylint: skip-file

import json
from typing import List, Any
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class JsonFileTokenizer():

    def __init__(self, jsonFile, tokenizer, forEval=False):

        self.jsonFile = jsonFile
        self.tokenizer = tokenizer
        self.labels = None
        self.forEval = forEval

        self.len_text = None
        self.max_len = 0

        self.sentences = None
        self.tokenized_sentences = []
        self.token_ids = []
        self.token_segments = []

        self.tensor_pairs = []
        self.padded_sequence_tensor = None
        self.segments_tensor = None

    def read_json(self):

        with open(self.jsonFile, 'rb') as f:
            jfile = json.load(f)

        sents = [sentence['content'] for sentence in jfile['sentences']]

        if self.forEval:
            self.labels = np.array([sentence['deletedInRound'] for sentence in jfile['sentences']])

        self.sentences = sents
        self.len_text = len(sents)

        return sents

    """
    Add special tokens for BERT
    """

    @staticmethod
    def add_special_tokens(sentence):

        sentence = '[CLS] ' + sentence + ' [SEP]'

        return sentence

    """
    Tokenize Single Sentence
    """

    def tokenize_single_sentence(self, sentence):

        sentenceForBert = JsonFileTokenizer.add_special_tokens(sentence)
        token_list = self.tokenizer.tokenize(sentenceForBert)

        return token_list

    """
    Tokenize Sentences by Iterating over them
    """

    def tokenize_text(self):
        self.read_json()
        for sent in self.sentences:
            tokenized_sentence = self.tokenize_single_sentence(sent)
            if len(tokenized_sentence) > self.max_len:
                self.max_len = len(tokenized_sentence)
            self.tokenized_sentences.append(tokenized_sentence)

        return self.tokenized_sentences

    def token_to_ids(self, output=False):
        self.tokenize_text()
        for token_list in self.tokenized_sentences:
            id_list = self.tokenizer.convert_tokens_to_ids(token_list)
            segment_list = [1] * len(id_list)
            self.token_ids.append(id_list)
            self.token_segments.append(segment_list)

        if output:
            return self.token_ids

    def prepare_for_single_inference(self, output=False):
        self.clear_state()
        self.token_to_ids()
        assert len(self.token_ids) == len(self.token_segments)

        for tokens, segments in zip(self.token_ids, self.token_segments):
            token_tensor = torch.tensor([tokens])
            segment_tensor = torch.tensor([segments])
            self.tensor_pairs.append((token_tensor, segment_tensor))

        if output:
            return self.tensor_pairs

    def prepare_for_batch_inference(self):
        self.prepare_for_single_inference()
        token_tensor_list = [x[0].T for x in self.tensor_pairs]
        self.padded_sequence_tensor = pad_sequence(token_tensor_list).T
        self.segments_tensor = torch.ones(self.padded_sequence_tensor.shape)

        return self.padded_sequence_tensor, self.segments_tensor

    def clear_state(self):

        self.len_text = None
        self.max_len = 0

        self.sentences = None
        self.tokenized_sentences = []
        self.token_ids = []
        self.token_segments = []

        self.tensor_pairs = []
        self.padded_sequence_tensor = None
        self.segments_tensor = None


def select_layer(bertOut: tuple, layers: List[int], return_cls: Any) -> np.ndarray:
    """
    Selects and averages layers from BERT output
    Parameters:
    bertOut: tuple
    Tuple containing output of 12 intermediate layers after feeding a document.
    layers: List of integers
    List that contains which layer to choose. max = 11, min = 0.
    return_cls: bool
    Whether to use CLS token embedding as sentence embedding instead of averaging token embeddings.
    Returns:
    numpy.ndarray (n_sentences, embedding_size) Embedding size if default to 768.
    """

    n_layers = len(layers)
    n_sentences = bertOut[0].shape[0]
    n_tokens = bertOut[0].shape[1]

    assert min(layers) > -1
    assert max(layers) < 12

    if return_cls:
        cls_matrix = np.zeros((n_layers, n_sentences, 768))
        l_ix = 0
        for l, layer in enumerate(bertOut):
            if l not in layers:
                continue
            else:
                l_ix = l_ix + 1
            for s, sentence in enumerate(layer):
                cls_tensor = sentence[0].numpy()
                cls_matrix[l_ix - 1, s, :] = cls_tensor
        layer_mean_cls = np.mean(cls_matrix, axis=0)
        return layer_mean_cls

    else:
        token_matrix = np.zeros((n_layers, n_sentences, n_tokens - 2, 768))
        for l, layer in enumerate(bertOut):
            l_ix = 0
            if l not in layers:
                continue
            else:
                l_ix = l_ix + 1
            for s, sentence in enumerate(layer):
                for t, token in enumerate(sentence[1:-1]):  # Exclude [CLS] and [SEP] embeddings
                    token_tensor = sentence[t].numpy()
                    token_matrix[l_ix - 1, s, t, :] = token_tensor

        tokenwise_mean = np.mean(token_matrix, axis=2)
        layer_mean_token = np.mean(tokenwise_mean, axis=0)
        return layer_mean_token


class NDCG():
    def __init__(self, k):
        self.k = k
        self.max_score = None
        self.summary_score = None

    def __call__(self, labels, summ_index):
        self.max_score = np.sum(sorted(labels)[::-1][:self.k])
        self.summary_score = np.sum(labels[summ_index])

        return self.summary_score / self.max_score


class ClusterEmbeddings(BaseEstimator, TransformerMixin):

    def __init__(self, k, random_state=None):
        super(ClusterEmbeddings, self).__init__()

        self.k = k
        self.random_state = random_state
        self.cluster_centers = None
        self.selected_sentence_indices = []

    def fit_transform(self, X):
        self._X = X
        assert self._X.shape[1] == 768

        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        kmeans.fit_transform(X)
        self.cluster_centers = kmeans.cluster_centers_
        cluster_df = pd.DataFrame(self.cluster_centers)

        euc_dist = euclidean_distances(self._X, cluster_df)
        for centroid in range(euc_dist.shape[1]):
            self.selected_sentence_indices.append(np.argmin(euc_dist[:, centroid]))

        return sorted(self.selected_sentence_indices)


class AnnotatedExtractiveSummarizer():
    """Run summarization and score on annotated data
    """

    def __init__(self, tokenizer, model, k=4, layers=[11], use_CLS_token=False, doEval=True, random_state=None,
                 verbose=False):
        super(AnnotatedExtractiveSummarizer, self).__init__()
        self.tokenizer = tokenizer
        self.doEval = doEval
        self.model = model
        self.layers = layers
        self.use_CLS_token = use_CLS_token
        self.k = k
        self.random_state = random_state
        self.verbose = verbose

    def summarize(self, jsonPath):
        self._jsonTokenizer = JsonFileTokenizer(jsonPath, self.tokenizer, forEval=self.doEval)
        self._tokens, self._segments = self._jsonTokenizer.prepare_for_batch_inference()

        self.model.eval()
        if self.verbose:
            print('Generating Embeddings...')
        with torch.no_grad():
            outputs = self.model(self._tokens[0], self._segments[0])
        self._twelve_layers = outputs[2][1:]

        self._sentence_embeddings = select_layer(self._twelve_layers, [11], return_cls=self.use_CLS_token)

        self._cluster_model = ClusterEmbeddings(self.k, self.random_state)
        self._selected_indices = self._cluster_model.fit_transform(self._sentence_embeddings)

        selected_sentences = np.array(self._jsonTokenizer.sentences)[self._selected_indices]
        return selected_sentences

    def score(self):

        if self.doEval:

            ndcg_score = NDCG(k=self.k)
            self._score = ndcg_score(self._jsonTokenizer.labels, self._selected_indices)

            return self._score

        else:

            raise Exception("Not in evaluation mode")
