import os
import sys
from pathlib import Path

THIS_DIR     = Path(__file__).resolve().parent
ROOT_DIR     = THIS_DIR.parents[1]
SIM_DIR      = ROOT_DIR / 'simulation'
TRAINING_DIR = ROOT_DIR / 'training'
RESULTS_DIR  = THIS_DIR / 'results'

for _p in (str(ROOT_DIR), str(SIM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(SIM_DIR)


def result_path(*parts):
    path = RESULTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
