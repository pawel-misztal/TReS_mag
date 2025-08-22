import os

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import argparse
from pathlib import Path
from types import SimpleNamespace
from helpersmag.initData import InitData
import json
from tqdm.auto import tqdm;
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from helpersmag.accuracy import calc_PLCC, calc_SRCC, calc_MAE, calc_CosineSim
from typing import Callable, List, Dict, Literal
from helpersmag.statskeys import INIT_DATA, BEST_MODEL_PATH
from helpersmag.trainingUtils import prepareDataset,setSeed,getDevice,prepareTransforms,prepareDataloader,prepareModel,prepareOptimizer,prepareScheluder,prepareLossFn,loadCheckpoint

def evaltestStep(
    model: nn.Module,
    dataLoader: DataLoader,
    device = 'cpu',
    repeats:int = 2):
  model.eval()

  targets = torch.zeros((0)).unsqueeze(1).cpu()
  preds = torch.zeros((0)).unsqueeze(1).cpu()

  output:torch.Tensor

  images = 0
  with torch.inference_mode():
    for i in tqdm(range(repeats)):
      images = 0
      for _, data in enumerate(dataLoader):
        image, label = data
        image = image.to(device)
        label:torch.Tensor = label.to(device)

        output, _, _ = model(image)
        targets = torch.cat((targets, label.detach().cpu()), dim=0)
        preds = torch.cat((preds, output.detach().cpu()),dim=0)
        images += len(label)

  preds = preds.view((-1,images)).mean(0)
  targets = targets.view((-1,images)).mean(0)


  print(  f'SRCC: {calc_SRCC(preds, targets):.4f}, '
          f'PLCC: {calc_PLCC(preds, targets):.4f}, '
          f'MAE: {calc_MAE(preds, targets):.4f}, ')


#   if(stats != None):
#     stats["eval_preds"].append(preds.tolist())
#     stats["eval_plcc"].append(calc_PLCC(preds, targets).item())
#     stats["eval_srcc"].append(calc_SRCC(preds, targets).item())
#     stats["eval_mae"].append(calc_MAE(preds, targets).item())

  return calc_SRCC(preds,targets),calc_PLCC(preds, targets), calc_MAE(preds, targets), preds
      
   

def loadBest(path:Path|str, model:nn.Module,   device:str):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(device)
    # if(checkpoint["scheluder"]):
    #     scheluder.load_state_dict(checkpoint["scheluder"])

def testModel(modelCheckpoint:Path,steps:int = 50, dataset:str = "clive", normalize:bool = False, seed:int = 2137, modelPath:str=None):

    if(modelCheckpoint != None):
        with open(modelCheckpoint, 'r') as fr:
            stats = json.load(fr)
            initData = InitData(**stats[INIT_DATA])


    setSeed(initData)
    device = getDevice()
    trainTransform , testTransform = prepareTransforms(initData)
    trainDataset, testDataset = prepareDataset(initData, trainTransform, testTransform)
    _, testDataloader = prepareDataloader(initData, trainDataset, testDataset)
    model = prepareModel(initData)
    # optimizer = prepareOptimizer(initData, model)
    # scheluder = prepareScheluder(initData, optimizer)
    # lossFn = prepareLossFn(initData)

    if(modelPath is None):
        modelPath = stats[BEST_MODEL_PATH]
    print("Loading checkpoint ", modelPath)
    loadBest(modelPath, model,device)


    model.to(device)
    srcc,plcc,mae,preds = evaltestStep(model, testDataloader, device, steps)

    statsPreds = {
       "init_data": stats[INIT_DATA],
       "eval_preds" : [preds.tolist()],
       "eval_plcc": [plcc.item()],
       "eval_srcc": [srcc.item()],
       "eval_mae": [mae.item()]
    }
    
    modelPath = modelPath.replace(".pth",".json")
    modelPath = modelPath.replace("checkpoints","manual")

    if(Path(modelPath).parent.exists() == False):
       Path(modelPath).parent.mkdir(parents=True,exist_ok=True)

    with open(modelPath, 'w') as fw:
        json.dump(statsPreds, fw, indent=2)
   


def parseArgs() :
    args = argparse.ArgumentParser()
    args.add_argument("--modelPath", "-mp" ,dest="modelPath", type=str,
                      default=None,
                      help="path to .json file")
    args.add_argument("--path", "-p" ,dest="path", type=str,
                      default=None,
                      help="path to .json file")
    args.add_argument("--name", "-n" ,dest="name", type=str,
                      default=None,
                      help="name of .json file")
    args.add_argument("--steps","-s", dest="steps",type=int,
                      default=50, help="number of repetitions for testing")
    args.add_argument("--dataset", "-d", dest="dataset", type=str,
                      default=None,
                      help="dataset, default '' means all", choices=["","clive","kadid10k"])
    args.add_argument("--normalize" ,dest="normalize", type=bool,
                      default=False,
                      help="normalize dataset")
    args.add_argument("--seed", dest="seed",type=int,
                      default=2137, help="seed for dataset")
    parsed_args = args.parse_args()
    return parsed_args.path, parsed_args.name, parsed_args.steps, parsed_args.dataset,parsed_args.normalize,parsed_args.seed, parsed_args.modelPath



if __name__ == "__main__":
    """
    python3 -m  helpersmag.testModel -h
    """
    path, name, steps, dataset, normalize, seed, modelWeightsPath = parseArgs()

    if(path):
        modelPath = path
    if(name):
        if(".json" not in name):
            name = name + ".json"
        modelPath = Path("/home/mrpaw/Documents/Projects/Python/PytorchTestRocm/magisterka/checkpoints") / name

    testModel(modelPath, steps, dataset, normalize, seed, modelWeightsPath)