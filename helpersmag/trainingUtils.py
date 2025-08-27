import os

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

from helpersmag.initData import InitData


from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from helpersmag.initData import InitData
import tres.impl_at1 as tres
import torch
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Tuple, List, Literal
import random
import numpy as np
import argparse

from typing import List, Tuple


def prepareDataset(initData:InitData, trainTransList:List[nn.Module], testTransList:List[nn.Module]) -> Tuple[Dataset, Dataset]:
    """
    returns [trainDataset, testDataset] 
    throws error if dataset is not valid
    """
    if(initData.dataset == 'live'):
        from Datasets.LiveDataset import LIVEDataset, LIVE_PATH
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = LIVEDataset(LIVE_PATH, True,trainTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        testDataset = LIVEDataset(LIVE_PATH, False,testTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        return trainDataset, testDataset
    
    if(initData.dataset == 'clive'):
        from Datasets.CLIVEDataset import CLIVEDataset, CLIVE_PATH
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = CLIVEDataset(CLIVE_PATH, True,trainTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        testDataset = CLIVEDataset(CLIVE_PATH, False,testTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        return trainDataset, testDataset

    if(initData.dataset == 'kadid10k'):
        from Datasets.Kadid10kDataset import Kadid10kDataset, KADID10K_PATH
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = Kadid10kDataset(KADID10K_PATH, True,trainTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        testDataset = Kadid10kDataset(KADID10K_PATH, False,testTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        return trainDataset, testDataset
    

    if(initData.dataset == 'biq2021'):
        from Datasets.BIQ2021Dataset import BIQ2021Dataset, BIQ2021_PATH
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = BIQ2021Dataset(BIQ2021_PATH, True,trainTrans,seed=initData.seed)
        testDataset = BIQ2021Dataset(BIQ2021_PATH, False,testTrans,seed=initData.seed)
        return trainDataset, testDataset
    
    if(initData.dataset == 'koniq10k'):
        from Datasets.Koniq10kDataset import Koniq10kData, KONIQ10K_PATH
        # if(trainTransList):
        #     trainTransList.insert(2,v2.Resize((512, 384)))
        #     testTransList.insert(2,v2.Resize((512, 384)))
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = Koniq10kData(KONIQ10K_PATH, True,trainTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        testDataset = Koniq10kData(KONIQ10K_PATH, False,testTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        return trainDataset, testDataset
    

    if(initData.dataset == 'zkoniq10k'):
        from Datasets.Koniq10kDataset import Koniq10kData, KONIQ10K_PATH
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = Koniq10kData(KONIQ10K_PATH, True,trainTrans,seed=initData.seed, normalize=initData.dataset_normalized,mos_z=True)
        testDataset = Koniq10kData(KONIQ10K_PATH, False,testTrans,seed=initData.seed, normalize=initData.dataset_normalized,mos_z=True)
        return trainDataset, testDataset

    if(initData.dataset == "bid"):
        from Datasets.BIDDataset import BIDDataset, BID_PATH
        if(trainTransList):
            trainTransList.insert(2,v2.Resize(size=None,max_size=512))
            testTransList.insert(0,v2.Resize(size=None,max_size=512))
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = BIDDataset(BID_PATH, True,trainTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        testDataset = BIDDataset(BID_PATH, False,testTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        return trainDataset, testDataset
    
    if(initData.dataset == 'tid2013'):
        from Datasets.TID2013Dataset import TID2013Dataset, TID2013_PATH
        trainTrans = v2.Compose(trainTransList) if trainTransList else None
        testTrans = v2.Compose(testTransList) if testTransList else None
        trainDataset = TID2013Dataset(TID2013_PATH, True,trainTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        testDataset = TID2013Dataset(TID2013_PATH, False,testTrans,seed=initData.seed, normalize=initData.dataset_normalized)
        return trainDataset, testDataset

    raise Exception(f"not supported dataset '{initData.dataset}'")

def prepareTransforms(initData:InitData) -> Tuple[List[nn.Module],List[nn.Module]]:
    trainTransform:List[nn.Module] = None
    testTransform:List[nn.Module] = None
    if(initData.train_transform == 'default'):
        trainTransform = [
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.RandomCrop((224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ]

    if(initData.test_transform == "default"):
        testTransform = [
            v2.RandomCrop((224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ]

    if(initData.train_transform == 'v2'):
        trainTransform = [
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            # v2.RandomCrop((224,224)),
            v2.RandomApply([v2.RandomRotation(degrees=(-25,25),expand=False)],0.5),
            v2.RandomResizedCrop(size=(224,224),scale=(0.1,1),ratio=(1,1)),
            v2.ToTensor(),
            v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ]

    if(initData.train_transform == 'rrc'):
        trainTransform = [
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            # v2.RandomCrop((224,224)),
            # v2.RandomApply(v2.RandomRotation(degrees=(-25,25),expand=False),0.5),
            v2.RandomResizedCrop(size=(224,224),scale=(0.2,1),ratio=(1,1)),
            v2.ToTensor(),
            v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ]


    if(trainTransform == None):
        raise Exception(f"train transform is invalidi '{initData.train_transform}'")
    
    if(testTransform == None):
        raise Exception(f"test transform is invalidi '{initData.test_transform}'")

    return trainTransform, testTransform

def prepareDataloader(initData:InitData,trainDataset:Dataset, testDataset:Dataset) -> Tuple[DataLoader, DataLoader]:
    batch_size = initData.batch_size
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return trainDataLoader, testDataLoader

def getDevice() -> Literal['cuda', 'cpu']:
    return "cuda" if torch.cuda.is_available() else 'cpu'

def setSeed(initData:InitData):
    seed = initData.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def prepareModel(initData:InitData) -> nn.Module:
    if(initData.model_class_name == "Tres"):
        model = tres.Tres(
            mhsa_dropout=initData.mhsa_dropout,
            mhsa_addPoseEveryLayer=initData.mhsa_add_pose_everyLayer,
            ffn_dropout=initData.ffn_dropout,
            normalizePosEncode=initData.normalize_pos_encode,
            normalizeBefore=initData.normalize_before,
            ffn_extraDropout=initData.ffn_extraDropout,
            ffn_size=initData.ffn_size,
            fc_last_dropout=initData.fc_last_dropout,
            fc_trans_dropout=initData.fc_trans_dropout,
            cnn_name=initData.cnn_model,
            init_xavier=initData.init_xavier,
            extraNormalizeAfter=initData.extraNormalizeAfterTrans,
            num_trans_encoders=initData.num_trans_encoders,
            one_more_linear=initData.one_more_linear,
            l2_pool_paper=initData.l2_pool_paper
        )

        return model

    raise Exception(f"'model_class_name': '{initData.model_class_name}' is invalid")

def prepareOptimizer(initData:InitData, model:nn.Module) -> torch.optim.Optimizer:
    if(initData.optimizer == "Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=initData.lr, weight_decay=initData.weight_decay)
        return optimizer
    if(initData.optimizer == "AdamW"):
        return torch.optim.AdamW(model.parameters(), lr=initData.lr, weight_decay=initData.weight_decay)
    raise Exception(f"provided optimizer type is invalid optimizer:'{initData.optimizer}'")

def prepareScheluder(initData:InitData, optimizer:torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    name = initData.lr_scheluder.get("name")
    if(not name):
        return None

    if(name == "StepLR"):
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=initData.lr_scheluder["step_size"],
                                               gamma=initData.lr_scheluder["gamma"])
    
    if(name == "CyclicLR"):
        return torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                 base_lr=initData.lr_scheluder["base_lr"], 
                                                 max_lr=initData.lr_scheluder["max_lr"],
                                                 step_size_up=initData.lr_scheluder["step_size_up"], 
                                                 step_size_down=initData.lr_scheluder["step_size_down"], 
                                                 mode=initData.lr_scheluder["mode"], 
                                                 gamma=(1-5/float(initData.epoch_count)))
    
    return None

def prepareLossFn(initData:InitData) -> nn.Module:
    if(initData.loss_fn == "L1Loss"):
        return nn.L1Loss()
    if(initData.loss_fn == "MSELoss"):
        return nn.MSELoss()
    if(initData.loss_fn == "HuberLoss"):
        return nn.HuberLoss()
    if(initData.loss_fn == "MAExDynamicMarginRankingLoss"):
        from tres.dynamicMarginLoss import MAExDynamicMarginRankingLoss
        return MAExDynamicMarginRankingLoss(initData.loss_weights["w_mae"],
                                            initData.loss_weights["w_dmrl"],
                                            initData.loss_weights["dmrl_alpha"],
                                            initData.loss_weights["dmrl_sort"])
    if(initData.loss_fn == "MAExDynamicTripletLoss"):
        from tres.dynamicTripletLoss import MAExDynamicTripletLoss
        return MAExDynamicTripletLoss(initData.loss_weights["w_mae"],
                                      initData.loss_weights["w_dtl"])
    
    if(initData.loss_fn == "PaperLoss"):
        from tres.paperLoss import PaperLoss
        return PaperLoss(b_coef_1=initData.loss_weights.get("b_coef_1", 0.5), #relative ranking, self consis
                         b_coef_2=initData.loss_weights.get("b_coef_2", 0.05), #relative ranking
                         b_coef_3=initData.loss_weights.get("b_coef_3", 1)) #self conis

    if(initData.loss_fn == "PaperLossDML"):
        from tres.paperLoss import PaperLossDML
        return PaperLossDML(b_coef_1=initData.loss_weights.get("b_coef_1", 0.5), #relative ranking, self consis
                         b_coef_2=initData.loss_weights.get("b_coef_2", 0.05), #relative ranking
                         b_coef_3=initData.loss_weights.get("b_coef_3", 1)) #self conis
    
    raise Exception(f"invalid loss_fn:'{initData.loss_fn}'")

def loadCheckpoint(path:Path|str, model:nn.Module, optimizer:torch.optim.Optimizer, scheluder:torch.optim.lr_scheduler.LRScheduler, device:str):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    if(checkpoint["scheluder"]):
        scheluder.load_state_dict(checkpoint["scheluder"])

def getSeed() -> int:
    parser = argparse.ArgumentParser(description="seed parser")
    parser.add_argument("--seed", "-s", dest="seed", type=int,default=2137,help="seed for initialize")

    parsed, unknown = parser.parse_known_args()
    seed = parsed.seed
    print("Starting with seed:",seed)
    return seed