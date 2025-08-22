import os
import gc
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import tres.impl_at1 as tres
import torch
from Datasets.CLIVEDataset import CLIVEDataset, CLIVE_PATH
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from helpersmag.trainer import trainLoop
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'

trans = v2.Compose([
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
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


trainDataset = CLIVEDataset(CLIVE_PATH, True,trans,seed=2137666, normalize=False)
testDataset = CLIVEDataset(CLIVE_PATH, False,transTest2,seed=2137666, normalize=False)


batch_size = 32
trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)



class InitData:
    def __init__(self,posEncodeEveryLayer:bool,lossFn:nn.Module,name:str,normalizePose:bool,normalizeBefore:bool):
        self.posEncodeEveryLayer:bool = posEncodeEveryLayer
        self.lossFn:nn.Module = lossFn
        self.name:str = name
        self.normalizePose:bool = normalizePose
        self.normalizeBefore:bool = normalizeBefore
    

tests = [
    InitData(
        posEncodeEveryLayer=False,
        lossFn=nn.BCEWithLogitsLoss(),
        normalizeBefore=False,
        normalizePose=False,
        name="default"
    ),
    InitData(
        posEncodeEveryLayer=False,
        lossFn=nn.BCEWithLogitsLoss(),
        normalizeBefore=False,
        normalizePose=True,
        name="s_np"
    ),
    InitData(
        posEncodeEveryLayer=False,
        lossFn=nn.BCEWithLogitsLoss(),
        normalizeBefore=True,
        normalizePose=False,
        name="s_nb"
    ),
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.BCEWithLogitsLoss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel"
    ),
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.BCEWithLogitsLoss(),
        normalizeBefore=True,
        normalizePose=True,
        name="s_all"
    ),
]

tests2 = [
    InitData(
        posEncodeEveryLayer=False,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="default2"
    ),
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel2"
    ),
]
tests3 = [
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.BCELoss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel3"
    ),
]

tests4 = [
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.BCEWithLogitsLoss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel4"
    ),
]

# usunięto normalizację z datasetu
tests5 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel5"
    ),
]

tests5_5 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel5_5"
    ),
]


# normalizacja z powrotem ale lr 100 razy większy
tests6 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel6"
    ),
]

tests7 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=True,
        normalizePose=True,
        name="s_peel7"
    ),
]

tests8 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel8"
    ),
]

tests9 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel9"
    ),
]

tests9_1 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel9_1"
    ),
]
tests9_2 = [ 
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.L1Loss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel9_2"
    ),
]

tests10 = [
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.BCEWithLogitsLoss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel10"
    ),
]


tests11 = [
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.HuberLoss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel11"
    ),
]

tests12 = [
    InitData(
        posEncodeEveryLayer=True,
        lossFn=nn.HuberLoss(),
        normalizeBefore=False,
        normalizePose=False,
        name="s_peel12"
    ),
]


steps = 150

for td in tests9_2:
    lossFn = td.lossFn

    path = Path("test_tres1_imgs")
    fileName = td.name + ".png"
    path = path / fileName

    model = tres.Tres(mhsa_dropout=0.5, 
                      ffn_dropout=0.5, 
                      normalizePosEncode=td.normalizePose, 
                      normalizeBefore=td.normalizeBefore, 
                      mhsa_addPoseEveryLayer=td.posEncodeEveryLayer)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=5e-4)


    lrScheluder = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)

    try:
        trainLoop(model, 
                  trainDataLoader, 
                  testDataLoader, 
                  optimizer, 
                  lossFn, 
                  steps, 
                  device, 
                  sheluder=lrScheluder, 
                  save=False,
                  evalRepeats=50,
                  evalEveryEpoch=10,
                  saveModelName=td.name)
    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


