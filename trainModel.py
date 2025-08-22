import os
import gc

import socket

hostname = socket.gethostname()

if hostname == 'mrpawlinux': 
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

from helpersmag.initData import InitData
from helpersmag.trainingUtils import prepareDataset, setSeed,getDevice,prepareDataloader,prepareLossFn,prepareModel,prepareOptimizer,prepareScheluder,prepareTransforms, loadCheckpoint
import tres.impl_at1 as tres
import torch
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from helpersmag.trainer import trainLoop
from helpersmag.checkpointArgs import parseCheckpointArgs
from pathlib import Path
from typing import Tuple, List, Literal
import random
import numpy as np
import traceback 



def cnnFreezer(unfreezeEpoch:int):
    isFreezed = False
    isUnfreezed = True

    def frezzerFunc(i:int, model:tres.Tres):
        nonlocal isFreezed, isUnfreezed

        if(isFreezed == False):
            isFreezed = True
            for p in model.cnn.parameters():
                p.requires_grad = False
            print("freezed")

        if(i >= unfreezeEpoch and isUnfreezed == True):
            isUnfreezed = False
            for p in model.cnn.parameters():
                p.requires_grad = True
            print("unfreezed")
            
    return frezzerFunc

def main(initData:InitData,mergeModelAfterEvalEpoch:int = None, test:bool = True):

    modelCheckpoint, jsonCheckpoint = parseCheckpointArgs()
    canloadCheckpoint = modelCheckpoint != None and jsonCheckpoint != None
    
    setSeed(initData)
    device = getDevice()
    trainTransform , testTransform = prepareTransforms(initData)
    trainDataset, testDataset = prepareDataset(initData, trainTransform, testTransform)
    trainDataloader, testDataloader = prepareDataloader(initData, trainDataset, testDataset)
    model = prepareModel(initData)
    optimizer = prepareOptimizer(initData, model)
    scheluder = prepareScheluder(initData, optimizer)
    lossFn = prepareLossFn(initData)

    if(canloadCheckpoint):
        print("Loading checkpoint ", modelCheckpoint)
        loadCheckpoint(modelCheckpoint, model,optimizer,scheluder,device)

    epochFunc = None
    if(initData.freeze_cnn_for_epochs != 0):
        epochFunc = cnnFreezer(initData.freeze_cnn_for_epochs)

    steps = initData.epoch_count
    try:
        trainLoop(model, 
                  trainDataloader, 
                  testDataloader, 
                  optimizer, 
                  lossFn, 
                  steps, 
                  device, 
                  sheluder=scheluder, 
                  save=initData.save_model,
                  evalRepeats=initData.eval_repeats,
                  evalEveryEpoch=initData.eval_every_epoch,
                  saveModelName=initData.getName(),
                  initData=initData,
                  statsCheckpointPath=jsonCheckpoint,
                  mergeModelAfterEvalEpoch=mergeModelAfterEvalEpoch,
                  epochFunc=epochFunc,
                  optimizerStepEvery=initData.optimizerStepEvery,
                  test=test)
    except Exception as e:
        print("Error: ",e)
        traceback.print_exc()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()


    