import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__) / '..' / '..').absolute())

from sadedegel.server.__main__ import app  # noqa # pylint: disable=unused-import, wrong-import-position
