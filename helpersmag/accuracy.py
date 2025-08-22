import torch
from torch import Tensor


def calc_SRCC(pred:Tensor, target:Tensor) -> Tensor:
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    assert len(pred) == len(target), "tensors need to have the same length"

    pred = pred.view(-1)
    target = target.view(-1)

    pred = pred.argsort().argsort().float()
    target = target.argsort().argsort().float()

    n = torch.tensor(pred.shape[0], dtype=torch.float32)
    diff = target - pred
    diff2 = diff.pow(2)

    l = 6 * (diff2).sum()
    m = n * (n**2 - 1)

    return 1 - ( l / m)

def calc_PLCC(pred:Tensor, target:Tensor) -> Tensor:
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    assert len(pred) == len(target), "tensors need to have the same length"

    #qi - suubjective (target)
    #qi_ - predicted 
    #qm - avg suubjective (target)
    #qm_ - avgg predicted 
    avg_pred = pred.mean()
    avg_target = target.mean()

    target_avgTaret = target - avg_target
    pred_avgPred = pred - avg_pred

    eps = 1e-12
    l = (target_avgTaret * pred_avgPred).sum() + eps
    m = torch.sqrt(target_avgTaret.pow(2).sum() * pred_avgPred.pow(2).sum() + eps)

    return l / m


def calc_CosineSim(pred:Tensor, target:Tensor) -> Tensor:
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    assert len(pred) == len(target), "tensors need to have the same length"

    # m = torch.max(pred.max(), target.max())
    # pred = pred / m
    # target = target / m

    return torch.nn.functional.cosine_similarity(pred,target, dim=0).item()


def calc_MAE(pred:Tensor, target:Tensor) -> Tensor:
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    assert len(pred) == len(target), "tensors need to have the same length"

    diff = pred - target

    return diff.abs().mean()