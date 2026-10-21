import os
from multiprocessing import Pool

import numpy as np
import RNA

W, L, U = 200, 150, 8
MARGIN = 300


def accessibility(seq):
    """(len(seq), 8) float16: P(the 1..8-mer ending at each position is unpaired), RNAplfold -W 200 -L 150 -u 8 -T 37."""
    s = seq.upper().replace("T", "U")
    w = min(W, len(s))
    a = np.array(RNA.pfl_fold_up(s, U, w, min(L, w)), dtype=np.float64)[1:, 1:]
    if np.isnan(a[:, 0]).any():
        raise RuntimeError(f"RNAplfold dropped positions of a {len(s)}-nt sequence")
    return np.nan_to_num(a).astype(np.float16)


def accessibility_band(seq, start, end):
    """accessibility(seq)[start:end], folding only MARGIN nt around it (exact because MARGIN >= W)."""
    lo, hi = max(0, start - MARGIN), min(len(seq), end + MARGIN)
    return accessibility(seq[lo:hi])[start - lo:end - lo]


def read_fasta(path):
    name, chunks = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def _fold_to(job):
    path, seq = job
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, acc=accessibility(seq))
    os.replace(tmp, path)


def fold_fasta(fasta, out, jobs=1, part=0, parts=1):
    """Fold every record of `fasta` into out/<id>.npz (key "acc"); resumable, split across `parts` array tasks."""
    os.makedirs(out, exist_ok=True)
    todo = ((os.path.join(out, f"{name}.npz"), seq) for i, (name, seq) in enumerate(read_fasta(fasta))
            if i % parts == part and not os.path.exists(os.path.join(out, f"{name}.npz")))
    with Pool(jobs) as pool:
        for _ in pool.imap_unordered(_fold_to, todo):
            pass
