"""FoldSplice-Tissue: splice-site probability and usage in heart, liver, brain and testis, 10 kb context."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L = 32
W = np.array([11] * 8 + [21] * 4 + [41] * 4)
AR = np.array([1] * 4 + [4] * 4 + [10] * 4 + [25] * 4)
CL = 2 * int(np.sum(AR * (W - 1)))
TISSUES = ["heart", "liver", "brain", "testis"]
PSPLICE = [1, 4, 7, 10]
USAGE = [2, 5, 8, 11]
CHANNELS = [f"{t}_{k}" for t in TISSUES for k in ("unspliced", "spliced", "usage")]


class ResBlock(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(l)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)
        self.bn2 = nn.BatchNorm1d(l)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        return x + self.conv2(torch.relu(self.bn2(out)))


class Net(nn.Module):
    def __init__(self, n_in=12):
        super().__init__()
        self.n_in = n_in
        self.conv1 = nn.Conv1d(n_in, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i, (w, ar) in enumerate(zip(W, AR)):
            self.resblocks.append(ResBlock(L, int(w), int(ar)))
            if (i + 1) % 4 == 0:
                self.convs.append(nn.Conv1d(L, L, 1))
        for k in range(1, 9):
            setattr(self, f"conv_last{k}", nn.Conv1d(L, 2 if k % 2 else 1, 1))

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.skip(conv)
        for i, block in enumerate(self.resblocks):
            conv = block(conv)
            if (i + 1) % 4 == 0:
                skip = skip + self.convs[i // 4](conv)
        skip = skip[:, :, CL // 2:-CL // 2]
        heads = [getattr(self, f"conv_last{k}")(skip) for k in range(1, 9)]
        return torch.cat([F.softmax(h, dim=1) if h.shape[1] == 2 else torch.sigmoid(h) for h in heads], 1)


def readouts(y, acc, don):
    """Exon-inclusion scores from outputs y (..., 12, positions), averaged over the four tissues."""
    def site(channels):
        return np.mean([0.5 * (y[..., c, acc] + y[..., c, don]) for c in channels], 0)
    return {"psplice": site(PSPLICE), "usage": site(USAGE)}
