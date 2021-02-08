import os
from symspellpy import SymSpell, Verbosity

class SpellingCorrector:
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


    def correct_compound(self, s):
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


    def correct_doc(self, doc):
        """Given Sadedegel Doc, correct inside text and return List[str] with the
        corrected text.
        (Not returning a new Doc directly as that causes circular import due to Doc import)

        Args:
            doc (:obj:`sadedegel.bblock.Doc`): Doc in which text will be corrected
        """

        res = []
        for s in doc:
            res.append(self.correct_compound(str(s)))


        return res
