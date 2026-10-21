import argparse
from importlib import import_module
from pathlib import Path

MODELS = ["site", "tissue"]
EVALS = ["pnas", "compensatory", "mapt", "bltp1"]


def genes(a):
    module = import_module(f"foldsplice.data.{a.model}")
    with open(a.out, "w") as f:
        for datafile in a.datafiles:
            for name, seq in module.genes(datafile):
                f.write(f">{name}\n{seq}\n")


def fold(a):
    from foldsplice.fold import fold_fasta
    fold_fasta(a.fasta, a.out, a.jobs, a.part, a.parts)


def build(a):
    module = import_module(f"foldsplice.data.{a.model}")
    print(module.build(a.datafiles, a.folds, a.out, *([a.exclude] if a.exclude else [])))


def train(a):
    from foldsplice.models import ARMS
    module = import_module(f"foldsplice.train.{a.model}")
    if a.model == "site":
        module.train(a.data, a.val, a.out, ARMS[a.arm], a.seed, a.epochs or 30)
    else:
        module.train(a.data, a.out, ARMS[a.arm], a.seed, a.epochs or (14 if a.head is None else 4),
                     a.head, a.init, a.workers)


def predict(a):
    from foldsplice.fold import read_fasta
    from foldsplice.models import family, load
    from foldsplice.predict import predict as run
    nets, channels = load(a.model, a.arm, a.seeds), family(a.model).CHANNELS
    with open(a.out, "w") as f:
        f.write("\t".join(["id", "position", "base", *channels]) + "\n")
        for name, seq in read_fasta(a.fasta):
            for i, row in enumerate(run(nets, seq.upper())):
                f.write("\t".join([name, str(i + 1), seq[i], *(f"{v:.4g}" for v in row)]) + "\n")


def prepare(a):
    import_module(f"foldsplice.evals.{a.eval}").prepare(a.raw, a.jobs)


def evaluate(a):
    module = import_module(f"foldsplice.evals.{a.eval}")
    out = Path(a.out) / a.eval
    out.mkdir(parents=True, exist_ok=True)
    raw = out / f"{a.model}.npz"
    extra = [Path(a.out) / "pnas" / f"{a.model}.npz"] if a.eval == "compensatory" else []
    if extra and not extra[0].exists():
        raise SystemExit(f"{extra[0]} missing: compensatory is calibrated on `foldsplice eval pnas --model {a.model}`")
    if a.eval == "pnas":
        module.predict(a.model, raw, a.seeds, a.limit)
    else:
        module.predict(a.model, raw, a.seeds)
    table = module.metrics(raw, *extra)
    table.to_csv(out / f"{a.model}.tsv", sep="\t", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    p = argparse.ArgumentParser(prog="foldsplice")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("genes", help="write folding input (gene bodies, transcript strand) from datafiles")
    s.add_argument("model", choices=MODELS)
    s.add_argument("out")
    s.add_argument("datafiles", nargs="+")
    s.set_defaults(fn=genes)

    s = sub.add_parser("fold", help="RNAplfold accessibility for every FASTA record -> <out>/<id>.npz")
    s.add_argument("fasta")
    s.add_argument("out")
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--part", type=int, default=0)
    s.add_argument("--parts", type=int, default=1)
    s.set_defaults(fn=fold)

    s = sub.add_parser("build", help="training windows: sequence + structure channels and labels")
    s.add_argument("model", choices=MODELS)
    s.add_argument("out")
    s.add_argument("datafiles", nargs="+")
    s.add_argument("--folds", required=True)
    s.add_argument("--exclude", nargs="*", help="gene-id lists to drop (tissue model)")
    s.set_defaults(fn=build)

    s = sub.add_parser("train", help="train one replicate")
    s.add_argument("model", choices=MODELS)
    s.add_argument("data")
    s.add_argument("out")
    s.add_argument("--val", help="validation dataset (site model)")
    s.add_argument("--arm", choices=["fold", "seq"], default="fold")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--epochs", type=int)
    s.add_argument("--head", type=int, help="tissue model stage 2: fine-tune this head (0-7)")
    s.add_argument("--init", help="tissue model stage 2: stage-1 checkpoint")
    s.add_argument("--workers", type=int, default=6)
    s.set_defaults(fn=train)

    s = sub.add_parser("predict", help="per-position predictions for FASTA sequences given on their transcript strand")
    s.add_argument("model", choices=MODELS)
    s.add_argument("fasta")
    s.add_argument("out")
    s.add_argument("--arm", choices=["fold", "seq"], default="fold")
    s.add_argument("--seeds", type=int, nargs="*")
    s.set_defaults(fn=predict)

    s = sub.add_parser("prepare", help="rebuild data/<eval> from the public sources (scripts/get_data.sh)")
    s.add_argument("eval", choices=EVALS)
    s.add_argument("--raw", default="raw")
    s.add_argument("--jobs", type=int, default=8)
    s.set_defaults(fn=prepare)

    s = sub.add_parser("eval", help="score an eval -> results/<eval>/<model>.{npz,tsv}")
    s.add_argument("eval", choices=EVALS)
    s.add_argument("--model", choices=MODELS, required=True)
    s.add_argument("--seeds", type=int, nargs="*")
    s.add_argument("--limit", type=int, help="pnas: first N exons only (quick check)")
    s.add_argument("--out", default="results")
    s.set_defaults(fn=evaluate)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
