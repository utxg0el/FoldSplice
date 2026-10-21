import os
from importlib import import_module
from pathlib import Path

import torch

ARMS = {"fold": 12, "seq": 4}
WEIGHTS = Path(os.environ.get("FOLDSPLICE_WEIGHTS", Path(__file__).resolve().parents[2] / "weights"))


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def family(name):
    return import_module(f"foldsplice.models.{name}")


def load(name, arm="fold", seeds=None, dev=None):
    """Trained replicates of model `name` ("site" or "tissue"); arm "fold" = sequence + structure, "seq" = sequence only."""
    paths = sorted((WEIGHTS / name / arm).glob("s*.pt"), key=lambda p: int(p.stem[1:]))
    if seeds is not None:
        paths = [p for p in paths if int(p.stem[1:]) in set(seeds)]
    if not paths:
        raise FileNotFoundError(f"no weights in {WEIGHTS / name / arm}")
    nets = []
    for p in paths:
        net = family(name).Net(ARMS[arm])
        net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        net.seed = int(p.stem[1:])
        nets.append(net.eval().to(dev or device()))
    return nets
