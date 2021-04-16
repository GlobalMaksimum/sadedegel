from ..bblock.util import deprecate

deprecate("[yellow]sadedegel.tokenize[/yellow] module is deprecated", (0, 21, 0),
          post_message="Use [yellow]sadedegel.sbd[/yellow] instead.")
from ..bblock.sbd import RegexpSentenceTokenizer, NLTKPunctTokenizer  # noqa: F401
from ..bblock.util import tr_lower, tr_upper, __tr_upper__, __tr_lower__  # noqa: F401

from .. import Doc, Sentences
