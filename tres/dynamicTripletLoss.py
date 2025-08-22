import torch
from torch import nn


class DynamicTripletLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.tripletLoss = torch.nn.TripletMarginLoss(p=1)
        self.tripletLoss2 = torch.nn.TripletMarginLoss(p=1)

    def forward(self,input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if(len(input) < 4):
            return torch.tensor(0, dtype=torch.float32)
        
        if(len(input) % 2 != 0):
            input = input[:-1,:]
            target = target[:-1,:]
        
        sortedI = target.argsort(dim=0)

        d_min = input[sortedI[0]].squeeze(1)
        dp_min_p = input[sortedI[1]].squeeze(1)
        d_max_p = input[sortedI[-1]].squeeze(1)

        d_max = input[sortedI[-1]].squeeze(1)
        dp_max_m = input[sortedI[-2]].squeeze(1)
        d_min_m = input[sortedI[0]].squeeze(1)

        margin1 = target[sortedI[-1]] - target[sortedI[1]]
        margin2 = target[sortedI[-2]] - target[sortedI[0]]

        self.tripletLoss.margin = margin1.item()
        self.tripletLoss2.margin = margin2.item()

        loss1 = self.tripletLoss.forward(d_min, dp_min_p, d_max_p)
        loss2 = self.tripletLoss2.forward(d_max, dp_max_m, d_min_m)

        orderLoss = torch.relu(input[sortedI[0]] - input[sortedI[-1]])
        return loss1 + loss2 + orderLoss
        
class MAExDynamicTripletLoss(nn.Module):
    def __init__(self,w_mae:float=1,w_dtl:float=1):
        super().__init__()
        self.w_mae = w_mae
        self.w_dtl = w_dtl

        self.mae = nn.L1Loss()
        self.dmrl = DynamicTripletLoss()

    def forward(self,input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        loss_mae = self.mae.forward(input, target)
        loss_dtl = self.dmrl.forward(input,target)
        return self.w_mae * loss_mae + self.w_dtl * loss_dtl
    

if __name__ == "__main__":
    loss = DynamicTripletLoss()
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
    