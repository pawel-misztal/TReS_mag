from helpersmag.utils import  printOvveride, saveModel
import torch
from torch import nn
from typing import List
from tqdm.auto import tqdm;
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from helpersmag.accuracy import calc_PLCC, calc_SRCC, calc_MAE, calc_CosineSim
from helpersmag.modelMerger import getEmptyModelLike, addTwoModels, divideModelWeights
from typing import Callable, List, Dict, Literal
import gc
from pathlib import Path
import numpy as np
import json
import traceback
import os
import inspect

from helpersmag.initData import InitData

def evaltestStep(
    model: nn.Module,
    dataLoader: DataLoader,
    stats: Dict[str,List],
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
        # output= model(image)
        targets = torch.cat((targets, label.detach().cpu()), dim=0)
        preds = torch.cat((preds, output.detach().cpu()),dim=0)
        images += len(label)

  preds = preds.view((-1,images)).mean(0)
  targets = targets.view((-1,images)).mean(0)


  print(  f'SRCC: {calc_SRCC(preds, targets):.4f}, '
          f'PLCC: {calc_PLCC(preds, targets):.4f}, '
          f'MAE: {calc_MAE(preds, targets):.4f}, ')

  print('', end='\n')

  if(stats != None):
    stats["eval_preds"].append(preds.tolist())
    stats["eval_plcc"].append(calc_PLCC(preds, targets).item())
    stats["eval_srcc"].append(calc_SRCC(preds, targets).item())
    stats["eval_mae"].append(calc_MAE(preds, targets).item())

  return calc_SRCC(preds,targets), preds

def testStep(
    model: nn.Module,
    dataLoader: DataLoader,
    lossFn: nn.Module,
    stats: Dict[str,List],
    device = 'cpu'):
  model.eval()

  targets = torch.zeros((0)).unsqueeze(1).cpu()
  preds = torch.zeros((0)).unsqueeze(1).cpu()

  avgLoss = 0

  testLen = len(dataLoader)
  logStep = testLen // 100
  logStep = max(logStep, 10)
  output:torch.Tensor

  loss_fn_num_of_args = len(inspect.signature(lossFn.forward).parameters)

  with torch.inference_mode():
    for i, data in enumerate(dataLoader):
      image, label = data
      image = image.to(device)
      label:torch.Tensor = label.to(device)

      # output, _, _ = model(image)
      # # output= model(image)
      # loss = lossFn(output, label)

      if loss_fn_num_of_args == 2:
        output, _, _= model(image)
        loss:torch.Tensor = lossFn(output, label)
      elif loss_fn_num_of_args == 7:
        output, cnn, trans= model(image)
        output2, cnn2, trans2 = model(image.flip(3))
        loss:torch.Tensor = lossFn(output, label, output2, cnn, cnn2, trans, trans2)
      else:
        raise Exception("Unsuported number of arguments: ", loss_fn_num_of_args)
    
      avgLoss += loss.item()
      targets = torch.cat((targets, label.detach().cpu()), dim=0)
      preds = torch.cat((preds, output.detach().cpu()),dim=0)


      if(i % logStep == 0 or i == testLen - 1):
        printOvveride(
          f'  Testing... {i+1}/{testLen}, '
          f'Avg Loss: {(avgLoss if i == 0 else avgLoss/i):.4f}, '
          f'SRCC: {calc_SRCC(preds, targets):.4f}, '
          f'PLCC: {calc_PLCC(preds, targets):.4f}, '
          f'MAE: {calc_MAE(preds, targets):.4f}, '
          f'COS: {calc_CosineSim(preds, targets):.4f}\t\t\t\t')
      
  avgLoss = avgLoss / len(dataLoader)
  stats["test_loss"].append(avgLoss)
  stats["test_plcc"].append(calc_PLCC(preds, targets).item())
  stats["test_srcc"].append(calc_SRCC(preds, targets).item())
  stats["test_mae"].append(calc_MAE(preds, targets).item())

  # print(f"Accuracy {calcAccuracy(trueCount, falseCount)}");
  print('', end='\n')

def trainStep(
    model: nn.Module,
    dataLoader:torch.utils.data.dataloader.DataLoader,
    optimizer:Optimizer,
    lossFn:nn.Module,
    stats: Dict[str,List],
    device = 'cpu',
    sheluder:torch.optim.lr_scheduler.LRScheduler = None,
    optimizerStepEvery:int = 1):
  model.train()

  targets = torch.zeros((0)).unsqueeze(1).cpu()
  preds = torch.zeros((0)).unsqueeze(1).cpu()
  avgLoss = 0

  trainLen = len(dataLoader)
  logStep = trainLen // 100
  logStep = max(logStep, 10)

  image:torch.Tensor
  label:torch.Tensor
  conv:torch.Tensor
  conv2:torch.Tensor
  trans:torch.Tensor
  trans2:torch.Tensor

  optimizer.zero_grad()
  dataLen = len(dataLoader)

  loss_fn_num_of_args = len(inspect.signature(lossFn.forward).parameters)

  for i, data in enumerate(dataLoader):
    image, label = data
    image = image.to(device)
    label = label.to(device)

    if loss_fn_num_of_args == 2:
      output, _, _= model(image)
      loss:torch.Tensor = lossFn(output, label)
    elif loss_fn_num_of_args == 7:
      output, cnn, trans= model(image)
      output2, cnn2, trans2 = model(image.flip(3))
      loss:torch.Tensor = lossFn(output, label, output2, cnn, cnn2, trans, trans2)
    else:
      raise Exception("Unsuported number of arguments: ", loss_fn_num_of_args)
    
    # output, conv, trans = model(image)
    # output = model(image)

    # output2, _, _ = model(image.flip(3))
    # output2, conv2, trans2 = model(image.flip(3))
    # selfLoss = nn.functional.l1_loss(conv, conv2.detach()) + nn.functional.l1_loss(trans, trans2.detach())
    # loss:torch.Tensor = lossFn.forward(output, label, output2, selfLoss)

    # loss2:torch.Tensor = lossFn.forward(output2, label)
    # loss_both = loss + loss2
    # loss_both.backward()
    # print(loss)
    loss.backward()


    targets = torch.cat((targets, label.detach().cpu()), dim=0)
    preds = torch.cat((preds, output.detach().cpu()),dim=0)

    if (i + 1) % optimizerStepEvery == 0 or i + 1 == dataLen:
      optimizer.step()
      optimizer.zero_grad()

    avgLoss += loss.item()

    # if(i % 500 == 0):
      # print(f"Epoch {i}, loss: {loss}");

    if(i % logStep == 0 or i == trainLen - 1):
      lr_text = f'lr {sheluder.get_last_lr()[0]:.4f}, ' if sheluder is not None else 'lr nn, '

      printOvveride(
        f'  Training... {i+1}/{trainLen}, '
        f'{lr_text}'
        f'Avg Loss: {(avgLoss if i == 0 else avgLoss/i):.4f}, '
        f'SRCC: {calc_SRCC(preds, targets):.4f}, '
        f'PLCC: {calc_PLCC(preds, targets):.4f}, '
        f'MAE: {calc_MAE(preds, targets):.4f}, '
        f'COS: {calc_CosineSim(preds, targets):.4f}\t\t\t\t')



  if(sheluder != None):
    sheluder.step()
  avgLoss = avgLoss / len(dataLoader)

  stats["train_loss"].append(avgLoss)
  stats["train_plcc"].append(calc_PLCC(preds, targets).item())
  stats["train_srcc"].append(calc_SRCC(preds, targets).item())
  stats["train_mae"].append(calc_MAE(preds, targets).item())

  print('', end='\n')

def saveStats(
    stats: Dict[str,List],
    modelName:str,
    initData:InitData = None):
  
  name = "default"
  if modelName != None:
    name = modelName

  if(initData != None):
    stats["init_data"] = initData.__dict__

  path = Path(f'checkpoints/{name}.json')
  if path.parent.exists() == False:
    path.parent.mkdir(parents=True, exist_ok=True)

  with open(path, 'w') as fw:
    json.dump(stats, fw, indent=2)


def saveBestModel(model:nn.Module, modelName:str, stats:Dict[str,any]):
  name = modelName
  path = Path(f'checkpoints/{name}_best.pth')
  stats['best_model_path'] = str(path.absolute())

  
  if path.parent.exists() == False:
    path.parent.mkdir(parents=True, exist_ok=True)

  torch.save(model.state_dict(), path)

def saveLastModel(model:nn.Module, modelName:str, stats:Dict[str,any], optimizer:torch.optim.Optimizer, scheluder:torch.optim.lr_scheduler.LRScheduler):
  name = modelName
  path = Path(f'checkpoints/{name}_last.pth')
  stats['last_model_path'] = str(path.absolute())

  checkpoint = {
    "model" : model.state_dict(),
    "optimizer" : optimizer.state_dict(),
    "scheluder" : scheluder.state_dict() if scheluder != None else {}
  }
  
  if path.parent.exists() == False:
    path.parent.mkdir(parents=True, exist_ok=True)

  torch.save(checkpoint, path)

def deleteLastModel(stats:Dict[str,any]):
  p = Path(stats['last_model_path'])

  if p.exists() and p.is_file():
    os.remove(p)
    stats['last_model_path'] = ""

def tryMergeModel(stats:Dict[str,List],model:nn.Module, modelName:str, saveAfterEval:int = None):
  if(saveAfterEval == None):
    return
  
  evalEpochCount = len(stats['eval_epoch'])
  if(evalEpochCount <= saveAfterEval):
    return
  
  print("merging at ", evalEpochCount)
  
  name = modelName
  path = Path(f'merged/{name}.pth')
  stats['merged_model_path'] = str(path.absolute())

  if path.parent.exists() == False:
    path.parent.mkdir(parents=True, exist_ok=True)

  device = next(model.parameters()).device
  cpuModel = model.to('cpu')
  cur_weights = cpuModel.state_dict()

  if(path.exists() == False):
    avg_weights = getEmptyModelLike(cur_weights)
  else:
    avg_weights = torch.load(path.absolute(), map_location='cpu',weights_only=False)

  avg_weights = addTwoModels(avg_weights, cur_weights)

  torch.save(avg_weights, path.absolute())
  model.to(device)
  del avg_weights
  del cur_weights
  gc.collect()

def tryNormalizeMergedModel(stats:Dict[str,List],model:nn.Module, modelName:str, saveAfterEval:int = None):
  if(saveAfterEval == None):
    return

  name = modelName
  path = Path(f'merged/{name}.pth')
  stats['merged_model_path'] = str(path.absolute())

  if path.parent.exists() == False:
    path.parent.mkdir(parents=True, exist_ok=True)
  
  if(path.exists() == False):
    return
  
  evalEpochCount = len(stats['eval_epoch'])
  div =  evalEpochCount - saveAfterEval

  print("div",div)
  
  avg_weights = torch.load(path.absolute(), map_location='cpu',weights_only=False)

  avg_weights = divideModelWeights(avg_weights, div)

  torch.save(avg_weights, path.absolute())
  del avg_weights
  gc.collect()

def trainLoop(
    model:nn.Module,
    trainDataLoader:DataLoader,
    testDataLoader:DataLoader,
    optimizer:torch.optim.Optimizer,
    lossFn:nn.Module,
    epochCount = 4,
    device = 'cpu',
    sheluder:torch.optim.lr_scheduler.LRScheduler = None,
    saveModelName:str = None,
    save:bool = True,
    epochFunc:Callable[[int,nn.Module], None] = None,
    evalRepeats = 50,
    evalEveryEpoch = 10,
    initData:InitData = None,
    statsCheckpointPath:Path|str=None,
    mergeModelAfterEvalEpoch:int = None,
    optimizerStepEvery:int = 1):

  stats:Dict[str,List] = {
    "init_data" : {},
    "best_model_path": "",
    "last_model_path": "",
    "merged_model_path": "",
    "test_loss" : [],
    "test_plcc" : [],
    "test_srcc" : [],
    "test_mae" : [],
    "train_loss" : [],
    "train_plcc" : [],
    "train_srcc" : [],
    "train_mae" : [],
    "eval_plcc" : [],
    "eval_srcc" : [],
    "eval_mae" : [],
    "eval_preds": [], # lists of preds from every eval
    "eval_epoch": []
  }

  model.to(device)

  bestSrcc = 0

  if(statsCheckpointPath != None):
    with open(statsCheckpointPath, 'r') as fr:
      stats = json.load(fr)
      bestSrcc = torch.tensor(stats['eval_srcc']).max().item()
  try:
    for i in tqdm(range(epochCount)):
      
      if(statsCheckpointPath != None):
        #skipping to next epoch

        if(i <= stats["eval_epoch"][-1]):
          print("skipping epoch ", i)
          continue

      if(epochFunc is not None):
        epochFunc(i,model)

      trainStep(model, trainDataLoader, optimizer, lossFn, stats, device, sheluder)
      testStep(model,testDataLoader, lossFn, stats, device)
      

      if(i % evalEveryEpoch == 0 and evalEveryEpoch != -1 or i + 1 == epochCount):
        saveLastModel(model, saveModelName, stats, optimizer, sheluder)
        srcc, _ = evaltestStep(model, testDataLoader, stats, device, evalRepeats)
        stats["eval_epoch"].append(i)  
        if(srcc > bestSrcc):
          bestSrcc = srcc
          if(save and bestSrcc):
            saveBestModel(model, saveModelName, stats)
        saveStats(stats, saveModelName, initData)

        tryMergeModel(stats, model, saveModelName,mergeModelAfterEvalEpoch)

  
    # evaltestStep(model, testDataLoader,stats, device, evalRepeats)
    # stats["eval_epoch"].append(epochCount)  
  except Exception as e:
    print("Error: ", e)
    traceback.print_exc()
    torch.cuda.empty_cache()
    gc.collect()
  finally:
    # todo: save stats 
    saveStats(stats, saveModelName, initData)
    tryNormalizeMergedModel(stats, model, saveModelName, mergeModelAfterEvalEpoch)
    print("Done :D")
    bestIndex = np.array(stats["eval_srcc"]).argmax()
    print( f"Best  srcc:{np.array(stats["eval_srcc"])[bestIndex]} plcc:{np.array(stats["eval_plcc"])[bestIndex]}")
    pass

print("maghelper")