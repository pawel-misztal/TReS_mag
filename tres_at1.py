import os
import gc
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import tres.impl_at1 as tres
import torch
from Datasets.CLIVEDataset import CLIVEDataset, CLIVE_PATH
from Datasets.Kadid10kDataset import Kadid10kDataset, KADID10K_PATH
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from helpersmag.trainer import trainLoop

device = "cuda" if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'

trans = v2.Compose([
    v2.RandomHorizontalFlip(0.25),
    v2.RandomVerticalFlip(0.25),
    v2.RandomCrop((224,224)),
    # v2.Resize((224,224)),
    v2.ToTensor(),
    v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])
transTest = v2.Compose([
    v2.Resize((224,224)),
    v2.ToTensor(),
    v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])
transTest2 = v2.Compose([
    v2.RandomCrop((224,224)),
    v2.ToTensor(),
    v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])


# trainDataset = CLIVEDataset(CLIVE_PATH, True,trans,seed=2137666)
# testDataset = CLIVEDataset(CLIVE_PATH, False,transTest2,seed=2137666)


trainDataset = Kadid10kDataset(KADID10K_PATH, True,trans)
testDataset = Kadid10kDataset(KADID10K_PATH, False,transTest2)


batch_size = 32
trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True,num_workers=2)
testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True,num_workers=2)

model = tres.Tres(mhsa_dropout=0.1, ffn_dropout=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
# lossFn = nn.MSELoss()
lossFn = nn.L1Loss()
# lossFn = tres.CustomLoss(1,0.1,1,0.5)
# lossFn = tres.CustomLoss2(w_mae=1, w_srcc=0.4, w_plcc=0.4)
# lossFn = nn.BCEWithLogitsLoss()
# lossFn = nn.BCELoss()
# lossFn = tres.CustomLoss3(w_hubert=0.2, w_bce=1)

steps = 25

# lrScheluder = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=2e-5, total_steps=steps, pct_start=0.2)
# lrScheluder = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-8, 1e-4,step_size_up=2, step_size_down=14, mode="exp_range", gamma=(1-5/float(steps)))
# lrScheluder = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,eta_min=1e-7)

lrScheluder = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

# for p in model.cnn.parameters():
#     p.requires_grad = False

# def onLoop(i:int, model:tres.Tres):
#     if(i == 10):
#         for p in model.cnn.parameters():
#             p.requires_grad = True

try:
    trainLoop(model, trainDataLoader, testDataLoader, optimizer, lossFn, steps, device, sheluder=lrScheluder, scheluderStepsPerEpoch=1, save=True, showImg=True)
except Exception as e:
    print(e)
    torch.cuda.empty_cache()
    gc.collect()