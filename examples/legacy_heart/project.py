import sys
from pathlib import Path as _ExamplesPath

_EXAMPLES = _ExamplesPath(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from pathlib import Path

from support import ProjectPath

# ProjectPathインスタンスを作成
project = ProjectPath(Path(__file__).parent)
