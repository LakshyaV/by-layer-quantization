import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fragmap.probes import build_corpus  # noqa: E402

build_corpus(Path(__file__).resolve().parents[1] / "data/probes")
