"""BLTP1 c.13476C>T (exon 79): stabilises an exonic stem, measured dPSI -50."""
from pathlib import Path

import numpy as np
import pandas as pd

from foldsplice.encode import SL
from foldsplice.evals.mapt import fold_all, read_fa, scores
from foldsplice.models import ARMS, family

DATA = Path(__file__).resolve().parents[2] / "data" / "bltp1"
START = 122152331
ACC, DON, POS, REF, ALT = 196236, 196386, 196318, "C", "T"
MEASURED_DPSI = -50.0


def sequences(gene):
    assert gene[POS] == REF and gene[ACC - 2:ACC] == "AG" and gene[DON + 1:DON + 3] == "GT"
    return {"wt": gene, "variant": gene[:POS] + ALT + gene[POS + 1:]}


def predict(name, out, seeds=None):
    seqs = sequences(read_fa(DATA / "gene.fa"))
    np.savez(out, model=name, **scores(name, seqs, np.load(DATA / "folds.npz"), ACC, DON, seeds))


def metrics(path):
    z = np.load(path)
    rows = [dict(model="measured", arm="-", readout="psi", dpsi=MEASURED_DPSI)]
    for arm in ARMS:
        for readout, v in family(str(z["model"])).readouts(z[arm], 0, 1).items():
            d = (v[:, 1] - v[:, 0]) * 100
            rows.append(dict(model=str(z["model"]), arm=arm, readout=readout, replicates=len(d), wt=v[:, 0].mean() * 100,
                             variant=v[:, 1].mean() * 100, dpsi=d.mean(), dpsi_sd=d.std(ddof=1), negative=(d < 0).mean()))
    return pd.DataFrame(rows)


def prepare(raw, jobs=2):
    DATA.mkdir(parents=True, exist_ok=True)
    gene = read_fa(Path(raw) / "BLTP1.fa")
    open(DATA / "gene.fa", "w").write(f">BLTP1 NM_001384125.1 chr4:{START}-{START + len(gene) - 1}(+)\n{gene}\n")
    np.savez_compressed(DATA / "folds.npz", **fold_all(sequences(gene), DON // SL, jobs))
