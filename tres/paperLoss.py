import torch
from torch import nn, Tensor
from tres.dynamicTripletLoss import DynamicTripletLoss


class PaperLoss(nn.Module):
    def __init__(self, b_coef_1:float = 0.5, b_coef_2:float = 0.05, b_coef_3:float = 1):
        super().__init__()

        self.b_coef_1 = b_coef_1
        self.b_coef_2 = b_coef_2
        self.b_coef_3 = b_coef_3

        self.q_loss = nn.L1Loss()
        self.dt_loss = DynamicTripletLoss()

    def forward(self, input1:Tensor, target:Tensor, input2:Tensor, cnn1:Tensor, cnn2:Tensor, trans1:Tensor, trans2:Tensor) -> Tensor:
        l_quality = self.q_loss.forward(input1, target)
        l_quality2 = self.q_loss.forward(input2, target)
        l_relative_ranking = self.dt_loss.forward(input1, target)
        l_relative_ranking2 = self.dt_loss.forward(input2, target)

        l_self_consis = self.q_loss.forward(cnn1,cnn2.detach()) + self.q_loss.forward(trans1,trans2.detach()) \
            + self.q_loss.forward(cnn2,cnn1.detach()) + self.q_loss.forward(trans2,trans1.detach()) \
            + self.b_coef_1 * (self.q_loss.forward(l_relative_ranking, l_relative_ranking2.detach()) + self.q_loss.forward(l_relative_ranking2, l_relative_ranking.detach()))

        return l_quality + l_quality2 + self.b_coef_2 * (l_relative_ranking + l_relative_ranking2) + self.b_coef_3 * l_self_consis