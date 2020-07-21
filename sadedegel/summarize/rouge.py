from ._summ import BasicSummarizer
from sadedegel.metrics import rouge1_score
from transformers import AutoTokenizer

class FirstK(BasicSummarizer):
    def __init__(self, k=3):
        self.k = k
        super().__init__()

    def _select(self, sents):

        if type(self.k) == int:
            limit = self.k
        else:
            limit = min(ceil(self.k * len(sents)), len(sents))

        for i, sent in enumerate(sents):
            if i < limit:
                yield sent

class RougeRawScorer(BasicSummarizer):
    """
        Rogue-1 raw scorer, which returns list in form of
        [[sent_idx, rouge score],..]
        in descending order of rouge_score and sent_idx being the index of the sentence
        in the original document.

        Used for auto-labelling important sentences and processes all sentences.

        Parameters
        ==========
        tokenizer: Tokenizer to be used for unigram generation
    """

    def __init__(self, tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")):
        super().__init__()

        self.tokenizer = tokenizer

    def _get_unigrams(self, sents: list):
        unigrams = []

        for s in sents:
            toks = self.tokenizer.tokenize(s)
            toks = [t.lower() for t in toks if t.isalpha()]

            unigrams += toks

        return unigrams


    def _select(self, sents: list):
        sents_idx_by_score = [[i,0] for i in range(len(sents))] # keep the original idx with it

        for i,s in enumerate(sents):
            all_sents_except_s = sents[:i] + sents[i+1:]

            hyp_grams = self._get_unigrams([s])
            ref_grams = self._get_unigrams(all_sents_except_s)
            print(hyp_grams, ref_grams)
            score = rouge1_score(hyp_grams, ref_grams, metric="recall")

            sents_idx_by_score[i][1] += score

        sents_idx_by_score.sort(key=lambda x: x[1], reverse=True)

        for i, sent in enumerate(sents_idx_by_score):
            yield sent

class RougeSummarizer(BasicSummarizer):
    def __init__(self, k=3, tokenizer=AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")):
        self.raw_scorer = RougeRawScorer(tokenizer)
        self.k = k

    def _select(self, sents: list):
        if type(self.k) == int:
            limit = self.k
        else:
            limit = min(ceil(self.k * len(sents)), len(sents))

        for i,x in enumerate(self.raw_scorer(sents)):
            if i < limit:
                sent_idx = x[0]
                yield sents[sent_idx]
