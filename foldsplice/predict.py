import numpy as np
import torch

from foldsplice.encode import inputs, structure


def run(nets, X, positions=None, batch=8):
    """Outputs (len(nets), len(X), channels, positions) of each net on inputs X (n, SL + CL, >= n_in)."""
    dev = next(nets[0].parameters()).device
    out = []
    with torch.no_grad():
        for net in nets:
            ys = []
            for i in range(0, len(X), batch):
                x = torch.from_numpy(np.ascontiguousarray(X[i:i + batch, :, :net.n_in], np.float32))
                y = net(x.permute(0, 2, 1).to(dev))
                ys.append((y if positions is None else y[:, :, positions]).cpu().numpy())
            out.append(np.concatenate(ys))
    return np.stack(out)


def predict(nets, seq, acc=None):
    """Replicate-mean outputs (len(seq), channels) for a pre-mRNA / gene body given on its transcript strand."""
    if acc is None and nets[0].n_in > 4:
        acc = structure(seq)
    y = run(nets, inputs(seq, acc)).mean(0)
    return y.transpose(0, 2, 1).reshape(-1, y.shape[1])[:len(seq)]
