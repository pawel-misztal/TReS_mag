import torch
from torch import nn

class DynamicMarginRankingLoss(torch.nn.Module):
    def __init__(self, alpha:float = 0, sort:bool = True, v2=False):
        super().__init__()

        self.alpha = alpha
        self.sort = sort
        self.v2 = v2
        self.marginLoss = torch.nn.MarginRankingLoss()
        self.marginLoss2 = torch.nn.MarginRankingLoss()

    def forward(self,input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if(len(input) < 4):
            return torch.tensor(0.0, device=input.device)
        
        if(len(input) % 2 != 0):
            input = input[:-1,:]
            target = target[:-1,:]
        
        sortedI = target.argsort(dim=0)
        if(self.sort):
            sortedTarget = target[sortedI,:].squeeze(2)
            sortedPred = input[sortedI,:].squeeze(2)
        else:
            sortedTarget = target
            sortedPred = input

        p02 = sortedPred[0::2,:]
        p13 = sortedPred[1::2,:]
        t02 = sortedTarget[0::2,:]
        t13 = sortedTarget[1::2,:]
        t = (t02 - t13).sign()
        tm = (t02 - t13).abs().mean()

        margin = tm * self.alpha
        self.marginLoss.margin = margin
        loss_01_12 = self.marginLoss.forward(p02,p13, t)

        _p13 = p13[0:-1,:]
        _p02 = p02[1:,:]
        _t13 = t13[0:-1,:]
        _t02 = t02[1:,:]

        if self.v2:
            _t = (_t13 - _t02).sign()
            _tm = (_t13 - _t02).abs().mean()
            _margin = _tm * self.alpha
            self.marginLoss2.margin = _margin
            _loss_13_02 = self.marginLoss2.forward(_p13,_p02,_t)
            loss2 = _loss_13_02

        else:
            p12 = p02[1:,:]
            p23 = p13[0:-1,:]
            t12 = t02[1:,:]
            t23 = t13[0:-1,:]
            tt = (t12 - t23).sign()
            ttm = (t12 - t23).abs().mean()
            margin = ttm * self.alpha
            self.marginLoss2.margin = margin
            loss_12_23 = self.marginLoss2.forward(p12,p23, tt)
            loss2 = loss_12_23

        return loss_01_12 + loss2
    
class MAExDynamicMarginRankingLoss(nn.Module):
    def __init__(self,w_mae:float=1,w_dmrl:float=1, dmrl_alpha:float=0.5,dmrl_sort:bool=True,v2=False):
        super().__init__()
        self.w_mae = w_mae
        self.w_dmrl = w_dmrl

        self.mae = nn.L1Loss()
        self.dmrl = DynamicMarginRankingLoss(dmrl_alpha,dmrl_sort,v2)    

    def forward(self,input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        loss_mae = self.mae.forward(input, target)
        loss_dmrl = self.dmrl.forward(input,target)
        return self.w_mae * loss_mae + self.w_dmrl * loss_dmrl
    
if __name__ == "__main__":
    loss = DynamicMarginRankingLoss(1)
    target = torch.arange(0,10, dtype=torch.float32).unsqueeze(1)
    preds = target
    print("loss1",loss.forward(preds,target))
    print(target)
    print(target.shape)

    noise = torch.rand(target.shape[0]).unsqueeze(1) - 0.5
    print("noise",noise)

    preds = torch.arange(10,0,step=-1,dtype=torch.float32).unsqueeze(1)
    print("loss2",loss.forward(preds,target))

    print(target.shape[0])
    preds = torch.zeros(preds.shape[0],dtype=torch.float32).unsqueeze(1)
    print("loss3",loss.forward(preds,target))

    preds = torch.arange(10,0,step=-1,dtype=torch.float32).unsqueeze(1) + noise *1
    print(preds)
    print("loss4",loss.forward(preds,target))

    preds = torch.arange(0,10, dtype=torch.float32).unsqueeze(1) + noise *1
    print(preds)
    print("loss5",loss.forward(preds,target))

    print("V2")
    loss = DynamicMarginRankingLoss(1,v2=True)
    
    preds = target
    print("loss1",loss.forward(preds,target))

    preds = torch.arange(10,0,step=-1,dtype=torch.float32).unsqueeze(1)
    print("loss2",loss.forward(preds,target))

    print(target.shape[0])
    preds = torch.zeros(preds.shape[0],dtype=torch.float32).unsqueeze(1)
    print("loss3",loss.forward(preds,target))

    preds = torch.arange(10,0,step=-1,dtype=torch.float32).unsqueeze(1) + noise *1
    print(preds)
    print("loss4",loss.forward(preds,target))

    preds = torch.arange(0,10, dtype=torch.float32).unsqueeze(1) + noise *1
    print(preds)
    print("loss5",loss.forward(preds,target))