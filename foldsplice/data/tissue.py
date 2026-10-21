"""FoldSplice-Tissue training windows from 4-tissue splice tables (datafile_<species>_<n>_<split>_all.h5) and gene folds."""
from pathlib import Path

import h5py
import numpy as np

from foldsplice.encode import CL, SL, inputs, n_windows

SPLICED = np.array([[1, 0], [0, 1], [0, 0]], np.float32)
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _stem(datafile):
    return Path(datafile).stem.removeprefix("datafile_")


def _body(f, i):
    body = _text(f["SEQ"][i]).upper()[CL // 2:-CL // 2]
    return body.translate(COMPLEMENT)[::-1] if _text(f["STRAND"][i]) == "-" else body


def genes(datafile):
    """(id, gene body on its transcript strand) for every gene without ambiguous bases."""
    with h5py.File(datafile, "r") as f:
        for i in range(len(f["SEQ"])):
            body = _body(f, i)
            if set(body) <= set("ACGT"):
                yield f"{_stem(datafile)}_{i}", body


def _spliced(cov):
    if cov == "":
        return 0
    cov = float(cov)
    return 3 if cov < -2.99 else 0 if cov < -1.99 else 1


def _usage(cov):
    if cov == "":
        return 0
    cov = float(cov)
    if cov < -2.99:
        return -1
    if cov < -1.99 or cov == 0:
        return 0
    if -1.01 < cov < -0.99:
        return -1
    return cov


def labels(strand, start, end, junctions):
    """Per-tissue spliced class (-1 unknown, 0, 1) and usage over the gene body, transcript orientation."""
    n = end - start + 1
    spliced, usage = -np.ones((4, n)), -np.ones((4, n))
    sites = junctions.split(";")[:-1]
    if sites:
        spliced[:], usage[:] = 0, 0
        for site in sites:
            coord, cov = site.split(":")
            coord, cov = int(coord), cov.split(",")
            if start <= coord <= end:
                i = coord - start if strand == "+" else end - coord
                for t in range(4):
                    spliced[t, i], usage[t, i] = _spliced(cov[t]), _usage(cov[t])
    spliced[(spliced == 3).any(1)] = -1
    return spliced, usage


def build(datafiles, folds, out, exclude=()):
    """X{i} (15000, 12) float16 and Y{i} (5000, 12) float32 for every window with a spliced site, datafiles in order."""
    drop = {line.strip() for path in exclude for line in open(path) if line.strip()}
    ctr = 0
    with h5py.File(out, "w") as h:
        for datafile in datafiles:
            stem = _stem(datafile)
            with h5py.File(datafile, "r") as f:
                for i in range(len(f["SEQ"])):
                    fold = Path(folds) / f"{stem}_{i}.npz"
                    if _text(f["NAME"][i]).split(".")[0] in drop or not fold.exists():
                        continue
                    body, strand = _body(f, i), _text(f["STRAND"][i])
                    start, end = int(_text(f["TX_START"][i])), int(_text(f["TX_END"][i]))
                    acc = np.load(fold)["acc"]
                    if not len(acc) == len(body) == end - start + 1:
                        raise ValueError(f"{stem}_{i}: fold, sequence and span differ in length")
                    X = inputs(body, acc).astype(np.float16)
                    spliced, usage = labels(strand, start, end, _text(f["JN_START"][i]))
                    pad = ((0, 0), (0, n_windows(len(body)) * SL - len(body)))
                    spliced = SPLICED[np.pad(spliced, pad, constant_values=-1).astype(int)]
                    usage = np.pad(usage, pad, constant_values=-1)[..., None]
                    for w in range(len(X)):
                        part = slice(SL * w, SL * (w + 1))
                        if spliced[:, part, 1].sum() < 1:
                            continue
                        h.create_dataset(f"X{ctr}", data=X[w])
                        h.create_dataset(f"Y{ctr}", data=np.concatenate(
                            [np.concatenate([spliced[t, part], usage[t, part]], 1) for t in range(4)], 1).astype(np.float32))
                        ctr += 1
    return ctr
