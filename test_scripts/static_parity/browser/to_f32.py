"""Convert the Edge probe's browser_<dtype>_<device>.json vectors into ../out/browser_<dtype>_<device>.f32.

    uv run python test_scripts/static_parity/browser/to_f32.py <dir holding the json files>
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out"
for f in sorted(Path(sys.argv[1]).glob("browser_*.json")):
    a = np.array(json.loads(f.read_text()), dtype="<f4")
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    a.tofile(OUT / f"{f.stem}.f32")
    print(f.stem, a.shape)
