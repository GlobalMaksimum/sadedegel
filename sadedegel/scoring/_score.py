from collections import Counter

class Rouge:
    _AVAILABLE_METRICS = ["f1", "recall", "precision"]
    def __init__(self, metric="f1", n=1):
        assert metric in Rouge._AVAILABLE_METRICS, "'%s' not a valid metric!".format(metric)
        assert n == 1, "Only Rouge-1 is supported (for now)."

        self.metric = metric


    def _get_unigrams(self, s: str) -> list:
        assert type(s)  == str

        s = s.lower()
        s_list = [c for c in s if c.isalpha() or c == ' ']
        s = "".join(s_list)

        unigrams = s.split(" ")
        unigrams = [s.strip() for s in unigrams if s != ""]

        return unigrams

    def _get_overlap_count(self, hyp_grams: list, ref_grams: list) -> int:
        hyp_grams = Counter(hyp_grams)
        ref_grams = Counter(ref_grams)

        overlap = (hyp_grams & ref_grams)
        overlap_count = len(list(overlap.elements())) # how many unigrams overlap, including dups

        return overlap_count

    def _get_recall(self, hyp_grams: list, ref_grams: list) -> float:
        overlap_count = self._get_overlap_count(hyp_grams, ref_grams)
        return overlap_count/len(ref_grams)

    def _get_precision(self, hyp_grams: list, ref_grams: list) -> float:
        overlap_count = self._get_overlap_count(hyp_grams, ref_grams)
        return overlap_count/len(hyp_grams)

    def _get_f1(self, hyp_grams: list, ref_grams: list) -> float:
        # recall and precision are calculated separately here as not to call
        # _get_overlap_count() twice
        overlap_count = self._get_overlap_count(hyp_grams, ref_grams)
        recall = overlap_count/len(ref_grams)
        precision = overlap_count/len(hyp_grams)

        f1 = (2*precision*recall)/(precision+recall)
        return f1

    def __call__(self, hyp: str, ref: str):
        hyp_grams = self._get_unigrams(hyp)
        ref_grams = self._get_unigrams(ref)

        if self.metric == "recall":
            return self._get_recall(hyp_grams, ref_grams)
        elif self.metric == "precision":
            return self._get_precision(hyp_grams, ref_grams)
        else:
            return self._get_f1(hyp_grams, ref_grams)
