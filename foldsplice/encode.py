import numpy as np

from foldsplice.fold import accessibility_band

SL, CL = 5000, 10000

_CODE = np.full(256, 4, np.uint8)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _CODE[ord(_b.lower())] = _i
_CODE[ord("U")] = _CODE[ord("u")] = 3
_ONEHOT = np.vstack([np.eye(4), np.zeros(4)]).astype(np.float32)


def onehot(seq):
    return _ONEHOT[_CODE[np.frombuffer(seq.encode(), np.uint8)]]


def n_windows(length):
    return -(-length // SL)


def structure(seq, windows=None):
    """Accessibility at every position the given windows read, zeros elsewhere."""
    windows = range(n_windows(len(seq))) if windows is None else windows
    lo = max(0, SL * min(windows) - CL // 2)
    hi = min(len(seq), SL * max(windows) + SL + CL // 2)
    acc = np.zeros((len(seq), 8), np.float16)
    acc[lo:hi] = accessibility_band(seq, lo, hi)
    return acc


def inputs(seq, acc=None, windows=None):
    """Model inputs (len(windows), SL + CL, 4 or 12) for `seq` read as a gene body on its transcript strand.

    Window w predicts seq[SL * w : SL * (w + 1)]; everything beyond the sequence ends is N with zero accessibility.
    """
    windows = range(n_windows(len(seq))) if windows is None else windows
    x = onehot(seq) if acc is None else np.concatenate([onehot(seq), acc.astype(np.float32)], 1)
    x = np.pad(x, ((CL // 2, CL // 2 + SL), (0, 0)))
    return np.stack([x[SL * w:SL * w + SL + CL] for w in windows])
