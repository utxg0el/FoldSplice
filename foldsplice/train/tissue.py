"""FoldSplice-Tissue training.

Stage 1 trains all 8 heads jointly (14 epochs, warm restarts at 2, 6, 14; keep the best validation epoch).
Stage 2 fine-tunes one head from the stage-1 checkpoint (4 epochs; keep the last).
Heads: 2t = P(splice) and 2t + 1 = usage for tissue t in heart, liver, brain, testis.
"""
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.utils import data

from foldsplice.models import device
from foldsplice.models.tissue import Net

BATCH = 12
SPLIT = [2, 1, 2, 1, 2, 1, 2, 1]


class Windows(data.Dataset):
    def __init__(self, path, n_in):
        self.path, self.n_in, self.h5, self.reads = path, n_in, None, 0
        with h5py.File(path, "r") as f:
            self.n = len(f) // (3 if "Z0" in f else 2)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.reads += 1
        if self.h5 is None or self.reads % 100000 == 0:
            if self.h5 is not None:
                self.h5.close()
            self.h5 = h5py.File(self.path, "r")
        x = torch.Tensor(self.h5[f"X{i}"][:, :self.n_in].T)
        return x, torch.Tensor(self.h5[f"Y{i}"][:].T)


def crossent(pred, true):
    return -torch.mean(true[:, 0] * torch.log(pred[:, 0] + 1e-10) + true[:, 1] * torch.log(pred[:, 1] + 1e-10))


def loss_fn(head=None):
    bce = nn.BCELoss()

    def head_loss(pred, true, h):
        if h % 2 == 0:
            return crossent(pred[h], true[h])
        known = true[h] >= 0
        return bce(pred[h][known], true[h][known])

    def loss(pred, true):
        pred, true = torch.split(pred, SPLIT, dim=1), torch.split(true, SPLIT, dim=1)
        heads = range(8) if head is None else [head]
        return sum(head_loss(pred, true, h) for h in heads)
    return loss


def train(dataset, out, n_in=12, seed=1, epochs=14, head=None, init=None, workers=6):
    """Stage 1 (head=None) or stage 2 (head=0..7, init=stage-1 checkpoint)."""
    dev, out = device(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed if head is None else 1000 * seed + head)
    ds = Windows(dataset, n_in)
    n_val = round(0.1 * len(ds))
    train_ds, val_ds = data.random_split(ds, (len(ds) - n_val, n_val))
    pin = dev.type == "cuda"
    train_dl = data.DataLoader(train_ds, BATCH, shuffle=True, num_workers=workers, pin_memory=pin)
    val_dl = data.DataLoader(val_ds, BATCH, shuffle=True, num_workers=workers, pin_memory=pin)
    net = Net(n_in)
    if init is not None:
        net.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
    net.to(dev)
    criterion = loss_fn(head)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 2 if head is None else 4, 2)
    tag = "" if head is None else f"head{head}_"
    best = float("inf")
    for epoch in range(epochs):
        net.train()
        for b, (x, y) in enumerate(train_dl):
            loss = criterion(net(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step(epoch + b / len(train_dl))
        net.eval()
        with torch.no_grad():
            val = sum(criterion(net(x.to(dev)), y.to(dev)).item() for x, y in val_dl) / len(val_dl)
        torch.save(net.state_dict(), out / f"{tag}epoch{epoch}.pt")
        if head is None and val < best:
            best = val
            torch.save(net.state_dict(), out / "best.pt")
        print(f"epoch {epoch}  val {val:.6f}", flush=True)
