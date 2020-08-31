from abc import ABC,abstractmethod
from ..bblock._base import _word_tokenizers


_global_config = {
    "Tokenizer": _word_tokenizers['bert']
}

def set_config(word_tokenizer_type: str):
    """Set global sadedegel configuration
    Parameters
    ----------
    word_tokenizer_type : str
        Select tokenizer for tokenizing Sentences object.

    See Also
    --------
    get_config: Retrieve current values of the global configuration
    """
    if word_tokenizer_type is not None:
        if word_tokenizer_type in ['bert','simple']:
            _global_config["Tokenizer"] = _word_tokenizers[word_tokenizer_type]
        else:
            raise NameError("Only available word tokenizer type options are [\"bert\", \"simple\"]")


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`
    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.
    See Also
    --------
    config_context: Context manager for global sadedegel configuration
    set_config: Set global sadedegel configuration
    """
    return _global_config.copy()
