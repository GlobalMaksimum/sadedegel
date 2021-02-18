import os, enum, string
from symspellpy import SymSpell, Verbosity
from loguru import logger

class SpellingCorrector:
    SPELLING_MODES = ["basic", "compound", "basic_compound"]
    DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "termfrequency_vocab.txt")
    PICKLED_DATA_PATH = os.path.join(os.path.expanduser("~"), ".sadedegel_data", "termfrequency_vocab.pickle")

    TURKISH_FLIPPED = str.maketrans({"i":"ı", "ı":"i", "I":"İ", "İ":"I",
                       "o":"ö", "ö":"o", "O":"Ö", "Ö":"O",
                       "c":"ç", "ç":"c", "C":"Ç", "Ç":"C",
                       "s":"ş", "ş":"s", "S":"Ş", "Ş":"S",
                       "g":"ğ", "ğ":"g", "G":"Ğ", "Ğ":"G",
                       "u":"ü", "ü":"u", "U":"Ü", "Ü":"U"})

    def __init__(self, max_dictionary_edit_distance=2, prefix_length=7, dict_path=None,
                 dont_use_pickled=False):
       
        """Wrapper class for SymSpell which acts as a bridge between
        Sadedegel and SymSpellPy.

        Parameters
        ----------
        max_dictionary_edit_distance : int
            Maximum edit distance for doing lookups.

        prefix_length : int
            The length of word prefixes used for spell checking.

        dict_path : str, default=None
            Path to a term frequency dictionary of words.
            If .txt file is passed, then it's assumed every row is in the format
            of:
                [word] [term frequency]
            if not a .txt file it's assumed to be a pickle file as saved by
            symspellpy (check symspellpy docs for more info)

            If not passed, loads default provided dictionary.

        dont_use_pickled : bool, default=False
            When dict_path == None, prevents loading the pre-generated
            pickled dictionary.

        """

        self.sym_spell = SymSpell(max_dictionary_edit_distance, prefix_length)
        self._max_edit_dist = max_dictionary_edit_distance
        self._dict_loaded = False # lazy loading for dict

        self.dict_path = dict_path
        if dict_path is None:
            if os.path.exists(self.PICKLED_DATA_PATH) and not dont_use_pickled:
                self.dict_path = self.PICKLED_DATA_PATH
            else:
                self.dict_path = self.DEFAULT_DATA_PATH

    def _load_dictionary(self):
        """Loads the dictionary specified by self.dict_path.
        Automatically called whenever a spelling is requested and
        dictionary isn't loaded yet.
        """


        logger.info(f"Loading term frequency dictionary from {self.dict_path}")

        if ".txt" in self.dict_path:
            is_ok = self.sym_spell.load_dictionary(self.dict_path, 0,1)
        else:
            is_ok = self.sym_spell.load_pickle(self.dict_path)

        if not is_ok:
            raise Exception(f"Could not load spelling dictionary! Dict path: {self.dict_path}")

        self._dict_loaded = True
        self._pickle_default_vocab()

    def _pickle_default_vocab(self):
        if self.dict_path == self.DEFAULT_DATA_PATH:
            logger.info("Pickling default loaded vocabulary for faster loading on next usage.")
            os.makedirs(os.path.split(self.PICKLED_DATA_PATH)[0], exist_ok=True)
            self.sym_spell.save_pickle(self.PICKLED_DATA_PATH)


    def _turkish_flip(self, w):
        return w.translate(self.TURKISH_FLIPPED)

    def _basic_with_punct(self, w):
        if not self._dict_loaded:
            self._load_dictionary()
        
        if len(w) == 0:
            return w
            
        stripped_w = w.translate(str.maketrans("", "", string.punctuation))
        if len(stripped_w) == 0:
            return w

        o = self.sym_spell.lookup(stripped_w,
            Verbosity.TOP,
            max_edit_distance=self._max_edit_dist,
            transfer_casing=True,
            include_unknown=False)

        o_flipped = self.sym_spell.lookup(self._turkish_flip(stripped_w),
                    Verbosity.TOP,
                    max_edit_distance=self._max_edit_dist,
                    transfer_casing=True,
                    include_unknown=False)


        if not o and not o_flipped: # neither can be found
            return w
        elif not o and o_flipped: # if flipped version was found, switch to it
            o = o_flipped
        elif o and o_flipped and o[0]._distance > o_flipped[0]._distance: # if both found get the one with smallest edit dist
            o = o_flipped

        word = o[0]._term
        if w[0].isupper():
            word = word[0].upper() + ''.join(word[1:])
        # find start punctuation
        # w is the original input word (along with any punctuation)
        start_idx = 0
        while w[start_idx] in string.punctuation and start_idx < len(w)-1:
            start_idx += 1
        start_punct = w[:start_idx]
        
        # find end punctuation
        end_idx = 1
        while w[-end_idx] in string.punctuation and end_idx < len(w):
            end_idx += 1
        if end_idx == 1:
            end_punct = ""
        else:
            end_punct = w[-end_idx+1:]

        return start_punct + word + end_punct

    # TODO: After a proper tokenizer implementation use List[Token] from the tokenizer instead
    def basic(self, s):
        """Given a string splits into words based on space and corrects the words
        by lookup.

        Parameters
        ----------
        s : str
            String to correct

        Returns
        -------
            str: Corrected string.
        """


        words = s.split(" ")
        words_fixed = [self._basic_with_punct(w.strip()) for w in words]

        return " ".join(words_fixed)


    def compound(self, s):
        """Given a phrase/sentence correct it and return another corrected string.
        Uses SymSpellPy's lookup_compound()

        Parameters
        ----------
        s : str
            String to correct

        Returns
        -------
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

        Parameters
        ----------
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
