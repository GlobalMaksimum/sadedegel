import torch
import numpy as np
from typing import List
from transformers import BertModel

class BertWrapper:
    """
        A wrapper class for BERT which enables easy device selection, finetuning,
        layer selection etc.
    """

    def __init__(self, pretrained_name: str = "dbmdz/bert-base-turkish-cased",
                 hidden_layers_picked: List[int] = [11], return_cls: bool = False,
                 device: str = "cpu"):

        """
            Parameters:
                pretrained_name: str
                    Name of the pretrained BERT model. Defaults to cased Turkish BERT.

                hidden_layers_picked: List[int]
                    List that contains which layers to choose. max = 11, min = 0.

                return_cls: bool
                    Whether to use CLS token embedding as sentence embedding instead of averaging token embeddings.

                device: str
                    Which device to run BERT on
        """

        self.model = BertModel.from_pretrained(pretrained_name, output_hidden_states=True)
        self.return_cls = return_cls
        self.hidden_layers_picked = hidden_layers_picked

        self.set_device(device)
        self.model.eval()

    def set_device(self, device: str):
        device = torch.device(device)
        self.model.to(device)

    def get_device(self) -> str:
        return next(self.model.parameters()).device

    def __call__(self, inp: torch.LongTensor, mask: torch.FloatTensor):
        """
            Parameters:
                inp: torch.LongTensor (n_sentences, sequence_length)
                    Input IDs of tokens

                mask: torch.FloatTensor (n_sentences, sequence_length)
                    Attention mask which indicates which tokens are padding
                    and are to be ignored

            Returns:
                numpy.ndarray (n_sentences, embedding_size) Embedding size if default to 768.
        """

        with torch.no_grad():
            outputs = self.model(inp, mask)

        twelve_layers = outputs[2][1:]
        return BertWrapper.select_layer(twelve_layers, self.hidden_layers_picked, return_cls=self.return_cls)



    @staticmethod
    def select_layer(bert_out: tuple, layers: List[int], return_cls: bool) -> np.ndarray:
        """Selects and averages layers from BERT output.

        Returns:
            numpy.ndarray (n_sentences, embedding_size) Embedding size if default to 768.

        """
        n_layers = len(layers)
        n_sentences = bert_out[0].shape[0]
        n_tokens = bert_out[0].shape[1]

        if not (min(layers) > -1 and max(layers) < 12):
            raise Exception(f"Value for layer should be in 0-11 range")

        if return_cls:
            cls_matrix = np.zeros((n_layers, n_sentences, 768))
            l_ix = 0
            for l, layer in enumerate(bert_out):
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
            for l, layer in enumerate(bert_out):
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
