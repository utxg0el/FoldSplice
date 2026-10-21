"""FoldSplice-Site: acceptor / donor / neither at every position, 10 kb context."""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

L = 32
W = np.array([11] * 8 + [21] * 4 + [41] * 4)
AR = np.array([1] * 4 + [4] * 4 + [10] * 4 + [25] * 4)
CL = 2 * int(np.sum(AR * (W - 1)))
CHANNELS = ["neither", "acceptor", "donor"]


class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l)
        self.batchnorm2 = nn.BatchNorm1d(l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=(w - 1) * ar // 2)

    def forward(self, x, skip):
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        return x + out, skip


class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, skip):
        return x, self.conv(x) + skip


class Net(nn.Module):
    def __init__(self, n_in=12):
        super().__init__()
        self.n_in = n_in
        self.initial_conv = nn.Conv1d(n_in, L, 1)
        self.initial_skip = Skip(L)
        self.residual_units = nn.ModuleList()
        for i, (w, ar) in enumerate(zip(W, AR)):
            self.residual_units.append(ResidualUnit(L, int(w), int(ar)))
            if (i + 1) % 4 == 0:
                self.residual_units.append(Skip(L))
        self.final_conv = nn.Conv1d(L, 3, 1)

    def forward(self, x):
        x, skip = self.initial_skip(self.initial_conv(x), 0)
        for unit in self.residual_units:
            x, skip = unit(x, skip)
        return F.softmax(self.final_conv(skip[:, :, CL // 2:-CL // 2]), dim=1)


def readouts(y, acc, don):
    """Exon-inclusion score from outputs y (..., 3, positions) at the exon's acceptor and donor."""
    return {"psi": 0.5 * (y[..., 1, acc] + y[..., 2, don])}
