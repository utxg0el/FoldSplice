<p align="center"><img src="assets/logo.png" alt="FoldSplice" width="420"></p>

FoldSplice predicts splicing from pre-mRNA sequence together with its RNA secondary structure.
Every position carries 4 sequence channels and 8 structure channels: the probability that the
1- to 8-nt stretch ending there is unpaired (RNAplfold, window 200, span 150, 37 °C). Each model
has a sequence-only twin trained the same way, so the contribution of structure is measured directly.

| model | predicts at every position | replicates (`fold` / `seq`) |
|---|---|---|
| `site` | acceptor, donor, neither | 5 / 5 |
| `tissue` | splice-site probability and usage in heart, liver, brain, testis | 30 / 30 |

Both read 5 kb of context on each side and predict 5 kb blocks.

## Install

```bash
pip install -e .
```

## Predict

Give the pre-mRNA (gene body) on its transcript strand; positions beyond its ends are treated as unknown.

```python
from foldsplice import load, predict

nets = load("site")          # the 5 sequence + structure replicates; load("site", "seq") for the twin
p = predict(nets, seq)       # (len(seq), 3): neither, acceptor, donor
```

```bash
foldsplice predict tissue genes.fa predictions.tsv
```

## Evaluate

Everything the evals need is in `data/`, except the PNAS exon folds (360 MB), which come from the release:

```bash
curl -L -o data/pnas/exon_folds.npy https://github.com/utxg0el/FoldSplice/releases/download/v0.1/exon_folds.npy
```

Any eval can also be rebuilt from the public sources with `scripts/get_data.sh raw` and `foldsplice prepare <eval> --raw raw`.

```bash
foldsplice eval pnas --model site            # results/pnas/site.{npz,tsv}
foldsplice eval compensatory --model site    # calibrated on the pnas run
foldsplice eval mapt --model site
foldsplice eval bltp1 --model site
```

| eval | data | reads out |
|---|---|---|
| `pnas` | 47,962 test exons of the Liao et al. 2023 minigene library | slope of calibrated error on exon MFE (0 = no structure blind spot) |
| `compensatory` | 5 stem-loop series: two stem-breaking singles and their re-pairing double | rescue (PSI points) and epistasis capture |
| `mapt` | 47 MAPT exon 10 variants, Kumar et al. 2022 | correlation of error with stem stability (0 = no blind spot) |
| `bltp1` | BLTP1 c.13476C>T, an exonic stem-stabilising variant | ΔPSI (measured −50) |

`notebooks/results.ipynb` draws the figures.

## Train

Windows are built from per-gene datafiles (sequence and labels) plus one fold per gene.

```bash
foldsplice genes site genes.fa datafile_train.h5 datafile_validation.h5 datafile_test.h5
foldsplice fold genes.fa folds --jobs 32                        # or --part i --parts N per array task
foldsplice build site dataset_train.h5 datafile_train.h5 --folds folds
foldsplice build site dataset_validation.h5 datafile_validation.h5 --folds folds
foldsplice train site dataset_train.h5 runs/site/fold_s1 --val dataset_validation.h5 --seed 1
foldsplice train site dataset_train.h5 runs/site/seq_s1 --val dataset_validation.h5 --seed 1 --arm seq
```

```bash
foldsplice genes tissue genes.fa datafile_*_all.h5
foldsplice fold genes.fa folds --jobs 32
foldsplice build tissue dataset_train.h5 datafile_Human_{1..5}_train_all.h5 datafile_Mouse_{1..6}_train_all.h5 \
    datafile_Macaque_{1..5}_train_all.h5 datafile_Rat_{2..5}_train_all.h5 --folds folds --exclude orthologs.txt
foldsplice train tissue dataset_train.h5 runs/tissue/fold_s1 --seed 1
for h in 0 1 2 3 4 5 6 7; do
  foldsplice train tissue dataset_train.h5 runs/tissue/fold_s1 --seed 1 --head $h --init runs/tissue/fold_s1/best.pt
done
```

| | `site` | `tissue` |
|---|---|---|
| labels | MANE v1.3 (GRCh38) canonical sites; test chr1, 3, 5, 7, 9 | RNA-seq splice-site usage, 4 tissues × human, macaque, mouse, rat |
| optimizer | AdamW, lr 1e-3, batch 12 | AdamW, lr 5e-4, batch 12 |
| schedule | cosine warm restarts every 5 epochs (to 1e-5), stepped per batch | cosine warm restarts after 2, 6, 14 epochs, stepped per batch |
| epochs | 30, keep best validation loss | 14, keep best validation loss; then 4 per head (8 heads) |
| loss | cross-entropy | cross-entropy (spliced) + binary cross-entropy (usage), summed over tissues |

Site datafiles (`pip install openspliceai`, MANE v1.3 GFF and GRCh38.p14 FASTA from NCBI):

```bash
openspliceai create-data --genome-fasta GCF_000001405.40_GRCh38.p14_genomic.fna --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff --output-dir . --remove-paralogs --min-identity 0.8 --min-coverage 0.8 --parse-type canonical --write-fasta --split-method human --canonical-only
```

Tissue datafiles, `orthologs.txt` and `paralogs.txt` (test set) come from the splice tables of
Zeng & Li 2022 ([preprocessing, step 2](https://github.com/tkzeng/Pangolin_train/tree/main/preprocessing)).

## Extending

A model is `models/<name>.py` (`Net(n_in)`, `CHANNELS`, `readouts(y, acc, don)`), `data/<name>.py` (`genes`, `build`)
and `train/<name>.py` (`train`), plus its name in `MODELS` in `foldsplice/__main__.py`; weights go in
`weights/<name>/{fold,seq}/s<seed>.pt`. An eval is `evals/<name>.py` (`prepare`, `predict`, `metrics`) with its
inputs in `data/<name>/`, plus its name in `EVALS`.

## Data

[Liao et al. 2023](https://doi.org/10.1073/pnas.2221165120) minigene library and stem loops, with predicted MFE from
[Liu et al.](https://github.com/regev-lab/splicing-interpretable-distillation);
MAPT exon 10 from [Kumar et al. 2022](https://doi.org/10.7554/eLife.73888);
BLTP1 from [this study](https://doi.org/10.1016/j.xhgg.2025.100521); loci from MANE v1.3 on GRCh38.

