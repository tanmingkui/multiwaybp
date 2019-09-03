import torch
import torch.nn as nn

__all__ = ["SVB", "BBN"]

# -----------------------------------------------------------------------------------------
# reference: https://arxiv.org/abs/1611.06013
def SVB(layer):
    eps = 5.0
    # for layer in m.modules():
    if isinstance(layer, nn.Conv2d):
        w_size = layer.weight.data.size()
        layer_weight = (layer.weight.data.view(w_size[0], -1)).cpu()
        U, S, V = torch.svd(layer_weight)
        S = S.clamp(1.0/(1+eps), 1+eps)
        layer_weight = torch.mm(torch.mm(U, torch.diag(S)), V.t())
        layer.weight.data.copy_(layer_weight.view(w_size[0], w_size[1], w_size[2],w_size[3]))


def BBN(layer):
    eps = 1.0
    # for layer in m.modules():
    if isinstance(layer, nn.BatchNorm2d):
        std = torch.sqrt(layer.running_var+layer.eps)
        alpha = torch.mean(layer.weight.data/std)
        low_bound = (std*alpha/(1+eps)).cpu()
        up_bound = (std*alpha*(1+eps)).cpu()
        layer_weight_cpu = layer.weight.data.cpu()
        layer_weight = layer_weight_cpu.numpy()
        layer_weight.clip(low_bound.numpy(), up_bound.numpy())
        layer.weight.data.copy_(torch.Tensor(layer_weight))

# -----------------------------------------------------------------------------------------

