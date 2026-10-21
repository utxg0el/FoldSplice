<p align="center"><img src="assets/logo.png" alt="FoldSplice" width="420"></p>

Splice-site prediction from sequence + RNA structure.
Abstract: [Folding It In: Structure-Aware Deep Splicing Models](https://newenglandcompbio.org/files/necb-2026-abstracts/A099.pdf) (NECB 2026).

Each position gets its one-hot base plus 8 accessibility tracks from RNAplfold (P(unpaired) for the 1-8 nt
stretch ending there, W=200, L=150, 37C). Every model has a sequence-only twin trained the same way, so you
can see what the structure actually adds.

- `site`: acceptor / donor / neither. 5 seeds (+5 seq-only)
- `tissue`: splice probability + usage in heart, liver, brain, testis. 30 seeds (+30 seq-only)

10 kb of context, 5 kb predicted per window.

## install

`pip install -e .`

## predict

Give it the pre-mRNA on its transcript strand. Anything past the ends is treated as N.

```python
from foldsplice import load, predict

nets = load("site")        # load("site", "seq") for the seq-only twin
p = predict(nets, seq)     # (len(seq), 3): neither, acceptor, donor
```

or from the shell: `foldsplice predict tissue genes.fa out.tsv`

## evals

Everything is in `data/`, grab those first:

```bash
curl -L -o data/pnas/exon_folds.npy https://github.com/utxg0el/FoldSplice/releases/download/v0.1/exon_folds.npy

foldsplice eval pnas --model site            # -> results/pnas/site.{npz,tsv}
foldsplice eval compensatory --model site    # calibrated on the pnas run, so do that first
foldsplice eval mapt --model site
foldsplice eval bltp1 --model site
```

- **pnas**: 47,962 test exons from the Liao 2023 minigene library. Slope of calibrated error vs exon MFE, 0 = no structure blind spot.
- **compensatory**: 5 stem loops, two stem-breaking singles and the double that re-pairs them. Rescue in PSI points, plus how much of the measured epistasis the model picks up.
- **mapt**: 47 MAPT exon 10 variants (Kumar 2022). Correlation of the error with stem stability, closer to 0 is better.
- **bltp1**: c.13476C>T, stabilises an exonic stem. Measured ΔPSI is -50.

Figures are in `notebooks/results.ipynb`. To rebuild the eval data from the public sources:
`scripts/get_data.sh raw` then `foldsplice prepare <eval> --raw raw`.

## training

Training data = per-gene datafiles (sequence + labels) + one fold per gene.

site:

```bash
foldsplice genes site genes.fa datafile_train.h5 datafile_validation.h5
foldsplice fold genes.fa folds --jobs 32        # --part i --parts N to split over an array job
for s in train validation; do foldsplice build site dataset_$s.h5 datafile_$s.h5 --folds folds; done
foldsplice train site dataset_train.h5 runs/site_s1 --val dataset_validation.h5 --seed 1    # --arm seq for the twin
```

The datafiles are MANE v1.3 on GRCh38 (test chroms 1, 3, 5, 7, 9), made with `openspliceai create-data`:

```bash
openspliceai create-data --genome-fasta GCF_000001405.40_GRCh38.p14_genomic.fna --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff --output-dir . --remove-paralogs --min-identity 0.8 --min-coverage 0.8 --parse-type canonical --write-fasta --split-method human --canonical-only
```

tissue:

```bash
foldsplice genes tissue genes.fa datafile_*_all.h5
foldsplice fold genes.fa folds --jobs 32
foldsplice build tissue dataset_train.h5 datafile_{Human_{1..5},Mouse_{1..6},Macaque_{1..5},Rat_{2..5}}_train_all.h5 --folds folds --exclude orthologs.txt
foldsplice train tissue dataset_train.h5 runs/tissue_s1 --seed 1      # stage 1, keeps best.pt
for h in {0..7}; do foldsplice train tissue dataset_train.h5 runs/tissue_s1 --seed 1 --head $h --init runs/tissue_s1/best.pt; done    # stage 2
```

Datafiles and `orthologs.txt` come from the Zeng & Li 2022 splice tables
([preprocessing, step 2](https://github.com/tkzeng/Pangolin_train/tree/main/preprocessing)).

Recipes:
- site: AdamW 1e-3, batch 12, cosine warm restarts every 5 epochs (min 1e-5), 30 epochs, best val loss
- tissue: AdamW 5e-4, batch 12, warm restarts at 2/6/14 epochs, 14 epochs + best val loss, then 4 epochs per head

## adding stuff

New model: `models/<name>.py` (`Net`, `CHANNELS`, `readouts`), `data/<name>.py`, `train/<name>.py`, and add it to
`MODELS` in `__main__.py`. Weights go in `weights/<name>/{fold,seq}/s<seed>.pt`.

New eval: `evals/<name>.py` (`prepare`, `predict`, `metrics`), inputs in `data/<name>/`, add it to `EVALS`.

## data

Liao et al. 2023 ([PNAS](https://doi.org/10.1073/pnas.2221165120)) minigene library and stem loops, predicted MFE
from [Liu et al.](https://github.com/regev-lab/splicing-interpretable-distillation), MAPT from Kumar et al. 2022
([eLife](https://doi.org/10.7554/eLife.73888)), BLTP1 from [here](https://doi.org/10.1016/j.xhgg.2025.100521).
Loci are MANE v1.3 / GRCh38.
