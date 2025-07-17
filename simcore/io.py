import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / "data"

def load_data():
    gen   = pd.read_csv(DATA_DIR / "gen_result.csv")
    dyn   = pd.read_csv(DATA_DIR / "dyn_params.csv")
    bus   = pd.read_csv(DATA_DIR / "bus_result.csv")
    B_gen = np.loadtxt(DATA_DIR / "B_gen.csv", delimiter=",")
    return gen, dyn, bus, B_gen
