from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def histogram(values: np.ndarray, outpath: Path, bins: int, xlabel: str) -> None:
    fig = plt.figure()
    plt.hist(np.asarray(values).ravel(), bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
