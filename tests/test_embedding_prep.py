from transformers import AutoTokenizer, BertModel
import numpy as np

from .context import PrepareDocEmb


def test_input_prep():

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased", output_hidden_states=False)
    sample_path = 'sadedegel/dataset/raw/amator-muyuz-profesyonel-miyiz-18592660.txt'

    with open(sample_path, 'r') as f:
        sample_text = f.read()

    prep = PrepareDocEmb(tokenizer, model)
    token_ids, token_type_ids = prep.prepare_input(sample_text)

    assert token_ids.shape[1] == int(434)
    assert token_type_ids.shape[0] == int(434)


def test_embedding_prep():

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    model = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased", output_hidden_states=False)
    sample_path = 'sadedegel/dataset/raw/amator-muyuz-profesyonel-miyiz-18592660.txt'

    with open(sample_path, 'r') as f:
        sample_text = f.read()

    prep = PrepareDocEmb(tokenizer, model)
    token_ids, _ = prep.prepare_input(sample_text)
    embs = prep.get_embeddings(sample_text)

    assert embs.shape[0] == np.where(token_ids.numpy()[0] == 2)[0].shape[0]
