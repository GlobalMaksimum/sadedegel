import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import json
from itertools import chain
import os
import sys
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer,BertModel
import torch

from sadedegel.tokenize import Doc

class PrepareDocEmb():
    
    """
    Get BERT Embeddings for each sentence in a text using Doc object.
    """
    
    def __init__(self,tokenizer,model):
        self.tokenizer = tokenizer
        self.model = model
        self._X = None
        
    def add_special_tokens(self,sentence):
        
        return '[CLS] ' + sentence + ' [SEP]'
    
    
    def prepare_input(self,text):
        
        document = Doc(text)
        sentences = [self.add_special_tokens(sentence) for sentence in document.sents]
        tokens = list(chain(*[self.tokenizer.tokenize(sentence) for sentence in sentences]))
        token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        
        pos_embs = []
        s = 0
        for tok in tokens:
            pos_embs.append(s)
            if tok=='[SEP]':
                s ^= 1
                
        pos_embs = torch.tensor(pos_embs)
                
        return token_ids, pos_embs
    
    def divide_text(self,tok,pos):
        
        """
        
        If text token length is larger than MAX_LEN = 512 BERT will have to discard sentences. 
        Instead document is divided without slicing border sentences in the middle.
        
        """
        
        tok_np = tok.numpy()[0]
        pos_np = pos.numpy()

        sep_ix = np.where(tok_np==3)[0] #SEP token id is 3 from AutoTokenizer.
        cls_ix = 0
        spans = []
        for i,sep in enumerate(sep_ix):
            sents_span = list(range(cls_ix,sep+1))
            if len(sents_span)>511:
                span = list(range(cls_ix,sep_ix[i-1]+1))
                cls_ix = sep_ix[i-1]+1
                spans.append(span)
        last_span = list(range(cls_ix,sep_ix[-1]+1))
        spans.append(last_span)

        divided_tok = []
        divided_pos = []
        for span in spans:
            divided_tok.append(tok_np[span])
            divided_pos.append(pos_np[span])

        return divided_tok,divided_pos
    
    
    def get_embeddings(self,text):
        
        inputs,type_ids = self.prepare_input(text)
        
        tok_divided,pos_divided = self.divide_text(inputs,type_ids)
        
        
        div = len(tok_divided)
        embs = []
        for (tok_np,pos_np) in zip(tok_divided,pos_divided):
            tok,pos = torch.tensor([tok_np]), torch.tensor(pos_np)

            self.model.eval()
            with torch.no_grad():
                last_hidden,_ = self.model(tok,token_type_ids = pos)

            cls_idx = np.where(tok[0]==2)[0]
            sent_embs = last_hidden[0][cls_idx].numpy()

            embs.append(sent_embs)
        embs = np.vstack(embs)
        
       
        return embs
            
        
    def get_all_embeddings(self,workdir):
        
        X = []
        for doc_name in tqdm(os.listdir(workdir)):
            try:
                with open(f'{workdir}/{doc_name}','r') as f:
                    text = f.read()
                
                embs = self.get_embeddings(text)
                

                X.append(embs)
                
            except:
                print(doc_name)

        X = np.vstack(X)
        self._X = X
        
        return X




