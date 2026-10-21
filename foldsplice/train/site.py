"""Site model training: AdamW 1e-3, batch 12, cosine warm restarts every 5 epochs, best val loss."""
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from foldsplice.models import device
from foldsplice.models.site import Net

BATCH = 12


def cross_entropy(pred, true):
    return -torch.mean(torch.sum(true * torch.log(pred + 1e-10), dim=1))


def shard(h5, k, n_in, shuffle):
    X = torch.from_numpy(h5[f"X{k}"][:, :, :n_in].astype(np.float32)).permute(0, 2, 1)
    Y = torch.from_numpy(h5[f"Y{k}"][0].astype(np.float32)).permute(0, 2, 1)
    return DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=shuffle, drop_last=True)


def n_shards(h5):
    return sum(k.startswith("X") for k in h5)


def validate(net, h5, n_in, seed, dev):
    """Loss on 1000 labelled validation windows, drawn with the same seeded sampling every epoch."""
    rng = np.random.RandomState(seed)
    labelled = []
    for k in rng.permutation(n_shards(h5)):
        y = h5[f"Y{k}"][0]
        labelled += [(k, i) for i in range(len(y) // BATCH * BATCH) if y[i].sum() >= 1]
    pick = sorted(labelled[j] for j in rng.permutation(len(labelled))[:1000])
    net.eval()
    total = 0.0
    with torch.no_grad():
        for k in sorted({k for k, _ in pick}):
            rows = [i for kk, i in pick if kk == k]
            for i in range(0, len(rows), BATCH):
                r = rows[i:i + BATCH]
                x = torch.from_numpy(h5[f"X{k}"][r][:, :, :n_in].astype(np.float32)).permute(0, 2, 1)
                y = torch.from_numpy(h5[f"Y{k}"][0, r].astype(np.float32)).permute(0, 2, 1)
                total += cross_entropy(net(x.to(dev)), y.to(dev)).item() * len(r)
    return total / len(pick)


def train(train_h5, val_h5, out, n_in=12, seed=1, epochs=30):
    dev, out = device(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    net = Net(n_in).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=1, eta_min=1e-5)
    with h5py.File(train_h5, "r") as tr, h5py.File(val_h5, "r") as va:
        order = np.random.RandomState(seed).permutation(n_shards(tr))  # same order every epoch, as in the original runs
        per_epoch = sum(len(tr[f"X{k}"]) // BATCH for k in order)
        step, best = 0, float("inf")
        for epoch in range(epochs):
            net.train()
            for k in order:
                for x, y in shard(tr, k, n_in, True):
                    loss = cross_entropy(net(x.to(dev)), y.to(dev))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step(step / per_epoch)
                    step += 1
            val = validate(net, va, n_in, seed, dev)
            torch.save(net.state_dict(), out / f"epoch{epoch}.pt")
            if val < best:
                best = val
                torch.save(net.state_dict(), out / "best.pt")
            print(f"epoch {epoch}  val {val:.6f}  best {best:.6f}", flush=True)
