import torch
import torch.nn as nn


class MLP(nn.Module):
    """84 -> 128 -> 64 -> 32 -> 1 feed-forward IDS (Section IV-D).

    batchnorm=False gives the DP-compatible network (21,249 parameters with 84
    inputs). Opacus cannot compute per-sample gradients through BatchNorm
    because BatchNorm mixes samples within a batch.
    """

    def __init__(self, d_in=84, hidden=(128, 64, 32), dropout=0.2, batchnorm=True):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- flatten / unflatten the *state* (params + BN buffers) for aggregation ----
def state_keys(model):
    return [k for k, v in model.state_dict().items() if v.dtype.is_floating_point]


def flatten_state(model, keys=None):
    sd = model.state_dict()
    keys = keys or state_keys(model)
    return torch.cat([sd[k].detach().reshape(-1).double().cpu() for k in keys]).numpy()


def load_flat_state(model, flat, keys=None):
    sd = model.state_dict()
    keys = keys or state_keys(model)
    off = 0
    flat_t = torch.as_tensor(flat)
    for k in keys:
        n = sd[k].numel()
        sd[k].copy_(flat_t[off:off + n].view_as(sd[k]).to(sd[k].dtype))
        off += n
    assert off == len(flat), "size mismatch while loading flat state"
