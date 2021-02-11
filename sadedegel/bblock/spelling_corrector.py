import os, enum, string
from symspellpy import SymSpell, Verbosity

class SpellingCorrector:
    SPELLING_MODES = ["basic", "compound", "basic_compound"]
    DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "termfrequency_vocab.txt")

    def __init__(self, max_dictionary_edit_distance=2, prefix_length=7, dict_path=None):
        """Wrapper class for SymSpell which acts as a bridge between
        Sadedegel and SymSpellPy.

        Args:
            max_dictionary_edit_distance (str):
            Maximum edit distance for doing lookups.

            prefix_length (int, optional):
            The length of word prefixes used for spell checking.

            dict_path (str, optional):
                Path to a term frequency dictionary of words.
                If .txt file is passed, then it's assumed every row is in the format
                of:
                    [word] [term frequency]
                if not a .txt file it's assumed to be a pickle file as saved by
                symspellpy (check symspellpy docs for more info)

                If not passed, loads default provided dictionary.

        """

        self.sym_spell = SymSpell(max_dictionary_edit_distance, prefix_length)
        self._max_edit_dist = max_dictionary_edit_distance
        self._dict_loaded = False # lazy loading for dict

        self.dict_path = dict_path
        if dict_path is None:
            self.dict_path = self.DEFAULT_DATA_PATH

    def _load_dictionary(self):
        """Loads the dictionary specified by self.dict_path.
        Automatically called whenever a spelling is requested and
        dictionary isn't loaded yet.
        """


        if ".txt" in self.dict_path:
            is_ok = self.sym_spell.load_dictionary(self.dict_path, 0,1)
        else:
            is_ok = self.sym_spell.load_pickle(self.dict_path)

        if not is_ok:
            raise Exception(f"Could not load spelling dictionary! Dict path: {self.dict_path}")

        self._dict_loaded = True


    def _basic_with_punct(self, w):
        if not self._dict_loaded:
            self._load_dictionary()

        o = self.sym_spell.lookup(w,
            Verbosity.CLOSEST,
            max_edit_distance=self._max_edit_dist,
            transfer_casing=True,
            include_unknown=True)

        if not o: return w

        word = o[0]._term
        if w[0].isupper():
            word = word[0].upper() + ''.join(word[1:])
        # find start punctuation
        start_idx = 0
        start_punct = ''
        while w[start_idx] in string.punctuation:
            start_punct += w[start_idx]
            if start_idx + 1 < len(w):
                start_idx += 1
            else:
                break
        # find end punctuation
        end_idx = 1
        end_punct = ''
        while w[-end_idx] in string.punctuation:
            end_punct += w[-end_idx]
            if end_idx - 1 > 0:
                end_idx -= 1
            else:
                break

        return start_punct + word + end_punct

    # TODO: After a proper tokenizer implementation use List[Token] from the tokenizer instead
    def basic(self, s):
        """Given a string splits into words based on space and corrects the words
        by lookup.

        Args:
            s (str): String to correct

        Returns:
            str: Corrected string.
        """


        words = s.split(" ")
        words_fixed = [self._basic_with_punct(w) for w in words]

        return " ".join(words_fixed)


    def compound(self, s):
        """Given a phrase/sentence correct it and return another corrected string.
        Uses SymSpellPy's lookup_compound()

        Args:
            s (str): String to correct

        Returns:
            str: Corrected string.
        """

        if not self._dict_loaded:
            self._load_dictionary()

        suggestions = self.sym_spell.lookup_compound(s, self._max_edit_dist, transfer_casing=True)
        return suggestions[0]._term


    def correct_doc(self, doc, spelling_mode="basic"):
        """Given Sadedegel Doc, correct inside text and return List[str] with the
        corrected text.
        (Not returning a new Doc directly as that causes circular import due to Doc import)

        Args:
            doc (:obj:`sadedegel.bblock.Doc`): Doc in which text will be corrected
        """

        assert spelling_mode in self.SPELLING_MODES, f"Spelling mode  {spelling_mode} not found!"

        res = []
        for s in doc:
            if spelling_mode == "basic":
                res.append(self.basic(str(s)))
            elif spelling_mode == "compound":
                res.append(self.compound(str(s)))
            elif spelling_mode == "basic_compound":
                res.append(self.compound(self.basic(str(s))))



        return res
