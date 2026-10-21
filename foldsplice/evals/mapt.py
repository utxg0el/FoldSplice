"""MAPT exon 10, 47 variants (Kumar 2022): correlation of the error with stem stability (strucMean), 0 = no blind spot."""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from foldsplice.encode import SL, inputs, structure
from foldsplice.models import ARMS, family, load
from foldsplice.predict import run

DATA = Path(__file__).resolve().parents[2] / "data" / "mapt"
START = 45894554
ACC, DON = 115756, 115848  # 0-based; old notebooks had 115755, off by one


def read_fa(path):
    return "".join(open(path).read().split("\n")[1:]).upper()


def sequences(gene, table):
    seqs = {"wt": gene}
    for r in table.fillna("").itertuples():
        assert gene[r.pos:r.pos + len(r.ref)] == r.ref and (len(r.ref) == len(r.alt) or r.pos > DON), r.name
        seqs[r.name] = gene[:r.pos] + r.alt + gene[r.pos + len(r.ref):]
    for s in seqs.values():
        assert s[ACC - 2:ACC] == "AG" and s[DON + 1:DON + 3] == "GT"
    return seqs


def scores(name, seqs, folds, acc, don, seeds=None):
    """Raw outputs {arm: (replicates, len(seqs), channels, 2)} at the exon's acceptor and donor."""
    w = don // SL
    assert acc // SL == w
    X = np.concatenate([inputs(s, folds[k], [w]) for k, s in seqs.items()])
    out = {}
    for arm in ARMS:
        nets = load(name, arm, seeds)
        out[arm], out[f"{arm}_seeds"] = run(nets, X, [acc - SL * w, don - SL * w]), np.array([n.seed for n in nets])
    return out


def predict(name, out, seeds=None):
    seqs = sequences(read_fa(DATA / "gene.fa"), pd.read_csv(DATA / "variants.tsv", sep="\t"))
    np.savez(out, model=name, **scores(name, seqs, np.load(DATA / "folds.npz"), ACC, DON, seeds))


def blind(pred, psi, struc):
    err = (pred - psi) * 100
    b, a = np.polyfit(pred, psi, 1)
    return dict(r=stats.pearsonr(pred, psi)[0], rho=stats.spearmanr(pred, psi)[0],
                blind_r=stats.pearsonr(err, struc)[0], blind_slope=np.polyfit(struc, err, 1)[0],
                blind_r_calibrated=stats.pearsonr(psi - (a + b * pred), struc)[0], mae=np.abs(err).mean())


def metrics(path):
    z, t = np.load(path), pd.read_csv(DATA / "variants.tsv", sep="\t")
    rows = [dict(model="spliceai (reference)", arm="-", readout="psi", **blind(t.spliceai, t.psi, t.struc))]
    for arm in ARMS:
        for readout, v in family(str(z["model"])).readouts(z[arm], 0, 1).items():
            per_seed = pd.DataFrame([blind(s[1:], t.psi, t.struc) for s in v])
            rows.append(dict(model=str(z["model"]), arm=arm, readout=readout, replicates=len(v),
                             **per_seed.mean(), **per_seed.std().add_suffix("_sd")))
    return pd.DataFrame(rows)


def fold_all(seqs, w, jobs):
    with Pool(jobs) as pool:
        return dict(zip(seqs, pool.starmap(structure, [(s, [w]) for s in seqs.values()])))


def prepare(raw, jobs=8):
    raw = Path(raw)
    DATA.mkdir(parents=True, exist_ok=True)
    gene = read_fa(raw / "MAPT.fa")
    open(DATA / "gene.fa", "w").write(f">MAPT NM_001377265.1 chr17:{START}-{START + len(gene) - 1}(+)\n{gene}\n")
    k = pd.read_csv(raw / "Kumar_2022_MAPT_SplicingPredictions.csv.gz")
    ref, alt = k["ref>alt"].str.split(">").str[0], k["ref>alt"].str.split(">").str[1]
    t = pd.DataFrame(dict(name=k.variant_name, pos=k.variant_locus.str.split(":").str[1].astype(int) - START,
                          ref=ref, alt=alt, psi=k.PSI, struc=k.strucMean,
                          spliceai=(k.spliceai_acceptor + k.spliceai_donor) / 2))
    t.to_csv(DATA / "variants.tsv", sep="\t", index=False)
    np.savez_compressed(DATA / "folds.npz", **fold_all(sequences(gene, t), DON // SL, jobs))
