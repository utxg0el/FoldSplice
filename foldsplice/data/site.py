"""FoldSplice-Site training windows from `openspliceai create-data` datafiles (SEQ, LABEL) and gene folds."""
from pathlib import Path

import h5py
import numpy as np

from foldsplice.encode import SL, inputs, n_windows

CHUNK = 100
LABELS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], np.int8)


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _stem(datafile):
    return Path(datafile).stem.removeprefix("datafile_")


def genes(datafile):
    """(id, gene body) for every gene without ambiguous bases; bodies are already on the transcript strand."""
    with h5py.File(datafile, "r") as f:
        for i in range(len(f["SEQ"])):
            seq = _text(f["SEQ"][i])
            if set(seq) <= set("ACGT"):
                yield f"{_stem(datafile)}_{i}", seq


def build(datafiles, folds, out):
    """X{k} (n, 15000, 12) float16 and Y{k} (1, n, 5000, 3) int8, 100 genes per chunk, folded genes only."""
    files = {d: h5py.File(d, "r") for d in datafiles}
    keep = [(d, i) for d, f in files.items() for i in range(len(f["SEQ"]))
            if (Path(folds) / f"{_stem(d)}_{i}.npz").exists() and set(_text(f["SEQ"][i])) <= set("ACGT")]
    with h5py.File(out, "w") as h:
        for c in range(0, len(keep), CHUNK):
            X, Y = [], []
            for d, i in keep[c:c + CHUNK]:
                seq, label = _text(files[d]["SEQ"][i]), _text(files[d]["LABEL"][i])
                acc = np.load(Path(folds) / f"{_stem(d)}_{i}.npz")["acc"]
                if not len(acc) == len(seq) == len(label):
                    raise ValueError(f"{_stem(d)}_{i}: fold, sequence and labels differ in length")
                X.append(inputs(seq, acc).astype(np.float16))
                y = np.full(n_windows(len(seq)) * SL, 3, np.int8)
                y[:len(label)] = np.frombuffer(label.encode(), np.uint8) - ord("0")
                Y.append(LABELS[y].reshape(-1, SL, 3))
            h.create_dataset(f"X{c // CHUNK}", data=np.concatenate(X))
            h.create_dataset(f"Y{c // CHUNK}", data=np.concatenate(Y)[None])
    for f in files.values():
        f.close()
    return len(keep)
