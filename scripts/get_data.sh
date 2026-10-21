#!/usr/bin/env bash
# Public sources behind data/: run, then `foldsplice prepare <eval> --raw raw` to rebuild data/<eval>.
set -euo pipefail
RAW=${1:-raw}
mkdir -p "$RAW"
DIST=https://raw.githubusercontent.com/regev-lab/splicing-interpretable-distillation/main/data
LIAO=https://raw.githubusercontent.com/regev-lab/interpretable-splicing-model/main

curl -fsSL -o "$RAW/Liao_2023_SplicingPredictions.csv.gz" "$DIST/Liao_2023_SplicingPredictions.csv.gz"
curl -fsSL -o "$RAW/Kumar_2022_MAPT_SplicingPredictions.csv.gz" "$DIST/Kumar_2022_MAPT_SplicingPredictions.csv.gz"
curl -fsSL -o "$RAW/Liao_2023_StemLoop_Predictions.csv" "$DIST/Liao_2023_StemLoop_Predictions.csv"
curl -fsSL -o "$RAW/Lib2.gbk" "$LIAO/fasta_files/Lib2.gbk"
curl -fsSL -o "$RAW/xTe_ES7_HeLa_ABC.pkl.gz" "$LIAO/data/xTe_ES7_HeLa_ABC.pkl.gz"

# MANE v1.3 gene bodies (GRCh38): MAPT NM_001377265.1, BLTP1 NM_001384125.1
for locus in "MAPT chr17 45894553 46028334" "BLTP1 chr4 122152330 122362752"; do
  set -- $locus
  curl -fsS "https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=$2;start=$3;end=$4" |
    python3 -c "import json, sys; print('>$1'); print(json.load(sys.stdin)['dna'].upper())" > "$RAW/$1.fa"
done
