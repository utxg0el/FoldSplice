"""Liao 2023 stem loops: two stem-breaking singles and the double that re-pairs them.

rescue = (single1 + single2) / 2 - (wt + double) / 2 in calibrated PSI points, averaged over series.
capture = mean model epistasis / mean measured epistasis, epistasis = L(double) - L(single1) - L(single2) + L(wt).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from foldsplice.evals import pnas
from foldsplice.models import ARMS, family

DATA = Path(__file__).resolve().parents[2] / "data" / "compensatory"
VARIANTS = ["wt", "single1", "single2", "double"]


def table():
    t = pd.read_csv(DATA / "stemloops.tsv", sep="\t")
    assert list(t.variant) == VARIANTS * (len(t) // 4)
    return t


def predict(name, out, seeds=None):
    bb = pnas.backbone()
    raw = pnas.scores(name, [pnas.construct(bb, s) for s in table().sequence], np.load(DATA / "folds.npy"), seeds)
    np.savez(out, model=name, **raw)


def rescue(p):
    return np.mean((p[:, 1] + p[:, 2]) / 2 - (p[:, 0] + p[:, 3]) / 2)


def epistasis(p):
    lg = logit(np.clip(p, 0.01, 0.99))
    return lg[:, 3] - lg[:, 1] - lg[:, 2] + lg[:, 0]


def metrics(path, pnas_path):
    z, ref = np.load(path), np.load(pnas_path)
    assert str(z["model"]) == str(ref["model"])
    readouts = family(str(z["model"])).readouts
    measured = table().psi.to_numpy().reshape(-1, 4)
    rows = [dict(model="measured", arm="-", readout="psi", rescue=rescue(measured * 100), capture=1.0)]
    for arm in ARMS:
        library = readouts(ref[arm], 0, 1)
        for readout, v in readouts(z[arm], 0, 1).items():
            p = v.mean(0)
            cal = pnas.calibrator(library[readout].mean(0), ref["psi"] * 100)(p)
            rows.append(dict(model=str(z["model"]), arm=arm, readout=readout, rescue=rescue(cal.reshape(-1, 4)),
                             capture=epistasis(p.reshape(-1, 4)).mean() / epistasis(measured).mean()))
    return pd.DataFrame(rows)


def prepare(raw, jobs=None):
    """Attach measured PSI to each construct, matched by its mutations."""
    t = table()
    csv = pd.read_csv(Path(raw) / "Liao_2023_StemLoop_Predictions.csv").set_index("sequence_id")
    for i, row in t.iterrows():
        wt = t.sequence[i - VARIANTS.index(row.variant)]
        muts = [(int(m[:-3]) - 3, m[-3], m[-1])  # labels count from 3 nt upstream of the exon
                for m in row.label.replace("U", "T").split()[1:]]
        assert sorted(muts) == [(j + 1, a, b) for j, (a, b) in enumerate(zip(wt, row.sequence)) if a != b]
        t.loc[i, "psi"] = csv.loc[row.label, "measured_psi"]
    t.to_csv(DATA / "stemloops.tsv", sep="\t", index=False)
    bb = pnas.backbone()
    np.save(DATA / "folds.npy", np.stack([pnas.fold_patch(pnas.construct(bb, s)) for s in t.sequence]))
