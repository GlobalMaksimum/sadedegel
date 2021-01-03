import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__) / '..' / '..').absolute())

from sadedegel import Doc  # noqa # pylint: disable=unused-import
from sadedegel.bblock.token import Token  # noqa # pylint: disable=unused-import
from sadedegel.bblock.vocabulary import Vocabulary  # noqa # pylint: disable=unused-import
