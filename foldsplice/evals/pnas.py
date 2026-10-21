"""Liao et al. 2023 minigene library: does a model over-predict inclusion of exons that fold tightly?

Each 70-nt test exon sits in the 8,612-nt Lib2 minigene; inclusion = 0.5 (P(acceptor) + P(donor)) at the cassette.
Predictions are mapped to measured PSI by an isotonic fit in logit space, and the blind spot is the slope on
predicted MFE in  error ~ 1 + MFE + PSI  (Liu et al.); 0 means no structure blind spot.
"""
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from foldsplice.encode import inputs
from foldsplice.fold import accessibility, accessibility_band
from foldsplice.models import ARMS, family, load
from foldsplice.predict import run

DATA = Path(__file__).resolve().parents[2] / "data" / "pnas"
SLOT = 3518, 3588
PATCH = 3318, 3788
SITES = [3515, 3590]
MEAS_EPS, MODEL_EPS = 1e-3, 1e-4


def construct(backbone, exon):
    return backbone[:SLOT[0]] + exon + backbone[SLOT[1]:]


def fold_patch(seq):
    return accessibility_band(seq, *PATCH)


def structure(backbone_fold, patch):
    acc = backbone_fold.copy()
    acc[PATCH[0]:PATCH[1]] = patch
    return acc


def backbone():
    seq = "".join(open(DATA / "lib2.fa").read().split("\n")[1:])
    assert len(seq) == 8612 and seq[3513:3518] == "AGGTT" and seq[3588:3593] == "CAGGT"
    return seq


def scores(name, constructs, folds, seeds=None, batch=64):
    """Raw outputs {arm: (replicates, n, channels, 2)} at the cassette acceptor and donor."""
    base = np.load(DATA / "backbone_fold.npy")
    nets = {arm: load(name, arm, seeds) for arm in ARMS}
    out = {arm: [] for arm in ARMS}
    for i in range(0, len(constructs), batch):
        X = np.concatenate([inputs(c, structure(base, f), [0])
                            for c, f in zip(constructs[i:i + batch], folds[i:i + batch])])
        for arm in ARMS:
            out[arm].append(run(nets[arm], X, SITES))
    return {arm: np.concatenate(v, 1) for arm, v in out.items()} | {
        f"{arm}_seeds": np.array([n.seed for n in nets[arm]]) for arm in ARMS}


def predict(name, out, seeds=None, limit=None):
    lib = pd.read_csv(DATA / "exons.tsv.gz", sep="\t").iloc[:limit]
    path = DATA / "exon_folds.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path}: download it (README) or run `foldsplice prepare pnas`")
    bb = backbone()
    raw = scores(name, [construct(bb, e) for e in lib.exon], np.load(path, mmap_mode="r"), seeds)
    np.savez(out, model=name, exon=lib.exon.to_numpy(str), psi=lib.psi, mfe=lib.mfe, spliceai=lib.spliceai, **raw)


def calibrator(pred, psi):
    """Isotonic map from logit(prediction) to logit(PSI); returns calibrated PSI (%) for new predictions."""
    def lg(p, eps):
        return logit(np.clip(p, eps, 1 - eps))
    iso = IsotonicRegression(out_of_bounds="clip").fit(lg(pred, MODEL_EPS), lg(psi / 100, MEAS_EPS))
    return lambda p: expit(iso.predict(lg(p, MODEL_EPS))) * 100


def blind_spot(pred, psi, mfe):
    """(slope on MFE of calibrated error controlling for PSI, Pearson r of calibrated prediction with PSI)."""
    cal = calibrator(pred, psi)(pred)
    X = np.column_stack([np.ones(len(psi)), mfe, psi])
    slope = np.linalg.lstsq(X, cal - psi, rcond=None)[0][1]
    return slope, np.corrcoef(cal, psi)[0, 1]


def metrics(path):
    z = np.load(path)
    psi, mfe, readouts = z["psi"] * 100, z["mfe"], family(str(z["model"])).readouts
    slope, r = blind_spot(z["spliceai"], psi, mfe)
    rows = [dict(model="spliceai (reference)", arm="-", readout="psi", n=len(psi), slope=slope, r=r)]
    for arm in ARMS:
        for readout, v in readouts(z[arm], 0, 1).items():
            slope, r = blind_spot(v.mean(0), psi, mfe)
            per_seed = [blind_spot(s, psi, mfe)[0] for s in v]
            rows.append(dict(model=str(z["model"]), arm=arm, readout=readout, n=len(psi), slope=slope, r=r,
                             slope_seed_mean=np.mean(per_seed), slope_seed_sd=np.std(per_seed, ddof=1),
                             replicates=len(v)))
    return pd.DataFrame(rows)


def _read_gbk(path):
    txt = open(path).read()
    return "".join(re.findall(r"[acgtnACGTN]", txt[txt.index("ORIGIN"):].split("\n", 1)[1])).upper()


def _fold_exon(args):
    bb, exon = args
    return fold_patch(construct(bb, exon))


def prepare(raw, jobs=8):
    """data/pnas from the public sources downloaded by scripts/get_data.sh."""
    import joblib
    raw = Path(raw)
    DATA.mkdir(parents=True, exist_ok=True)
    bb = _read_gbk(raw / "Lib2.gbk")
    open(DATA / "lib2.fa", "w").write(f">Lib2\n{bb}\n")
    bb = backbone()
    test = {"".join("ACGT"[i] for i in w.argmax(1))[10:80] for w in joblib.load(raw / "xTe_ES7_HeLa_ABC.pkl.gz")[0]}
    lib = pd.read_csv(raw / "Liao_2023_SplicingPredictions.csv.gz",
                      usecols=["exon", "PSI", "predicted_MFE", "spliceai_avg"])
    lib = lib[lib.exon.isin(test)].rename(columns={"PSI": "psi", "predicted_MFE": "mfe", "spliceai_avg": "spliceai"})
    assert lib.exon.is_unique and len(lib) == len(test)
    lib.to_csv(DATA / "exons.tsv.gz", sep="\t", index=False)
    np.save(DATA / "backbone_fold.npy", accessibility(construct(bb, lib.exon.iloc[0])))
    with Pool(jobs) as pool:
        np.save(DATA / "exon_folds.npy", np.stack(pool.map(_fold_exon, ((bb, e) for e in lib.exon), chunksize=64)))
