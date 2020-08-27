from abc import ABC,abstractmethod
from typing import List, Union, Tuple
from transformers import AutoTokenizer 
from ._core import word_tokenize

def get_text(X: str) -> str:

    if type(X) == str:
        return X
    else:
        return str(X)
    

class BaseTokenizer(ABC):

    @abstractmethod
    def _tokenize(self,text: str) -> List[str]:
        pass

    def tokenize(self, text: str) -> List[str]:

        return self._tokenize(text)

    def __call__(self,sentence: str) -> List[str]:
        txt = get_text(sentence)
        return self.tokenize(txt)

    @abstractmethod
    def __str__(self) -> str:
        pass

class BertTokenizer(BaseTokenizer):

    def __init__(self, keep_special_tokens=True):
        super(BertTokenizer,self).__init__
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.special = None
        self.tokens = None
        self.token_type_ids = None
        self.keep_special_tokens = keep_special_tokens

    def add_special_tokens(self,text: str) -> str:
        
        return "[CLS] " + text + " [SEP]"

    def filter_special_tokens(self, token_list: Union[List[int],List[str]]) -> Union[List[int],List[str]]:
        
        return token_list[1:-1]
    
    def convert_tokens_to_ids(self, token_list: List[str]) -> List[str]:
        
        return self.tokenizer.convert_tokens_to_ids(token_list)

    def  _tokenize(self, text: str) -> Tuple[List[str], List[int]]:
        self.special = self.add_special_tokens(text)
        tokens = self.tokenizer.tokenize(self.special)
        ids = self.convert_tokens_to_ids(tokens)
        
        if not self.keep_special_tokens:
            self.tokens = self.filter_special_tokens(tokens)
            self.token_type_ids = self.filter_special_tokens(ids)
        else:
            self.tokens = tokens
            self.token_type_ids = ids

        return self.tokens, self.token_type_ids

    def __str__(self):
        return "bert"

class SimpleTokenizer(BaseTokenizer):
    def __init__(self):
        super(SimpleTokenizer,self).__init__
        self.tokenizer = word_tokenize
        self.tokens = None
        self.token_type_ids = None

    def _tokenize(self, text: str) -> List[str]:
        self.tokens = self.tokenizer(text)
        self.token_type_ids = [0]*len(self.tokens)
        return self.tokens, self.token_type_ids

    def __str__(self) -> str:
        return "simple"

_word_tokenizers = {'bert':BertTokenizer(),
                    'simple':SimpleTokenizer()}