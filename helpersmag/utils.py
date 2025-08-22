import torch
from typing import Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
import pickle
from pathlib import Path

def printOvveride(values):
  print('',end='\r',flush=True)
  print(values, end='',flush=True)

def countTrueFalse(y_test, y_train) -> Tuple[int,int]:
  ''' return: true count, false count '''
  y_test = y_test.to("cpu")
  y_train = y_train.to("cpu")
  guess = torch.argmax(y_test, dim=1);
  same = torch.eq(guess, y_train).sum()
  # print(same);

  return [same, len(y_test) - same];

def calcAccuracy(trueCount, falseCount) -> float:
  return (trueCount / (trueCount + falseCount)) * 100;



def plotRes(epochCount,trainLoss,testLoss,accuracy,show=True,savePath:Path=None):
  plt.figure(figsize=(6,6));
  plt.plot(epochCount,trainLoss, label="train loss");
  plt.plot(epochCount,testLoss, label="test loss");
  plt.legend()
  if(show):
    plt.show()

  plt.figure(figsize=(6,6));
  plt.plot(epochCount,torch.asarray(accuracy), label="accuracy ");
  plt.legend()
  if(show):
    plt.show()
  if(savePath != None):
    if(savePath.parent.exists() == False):
      savePath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savePath)


def saveModel(epoch:int,model:nn.Module, modelName:str = None):
  name = model.__class__.__name__
  if modelName != None:
    name = modelName
  path = Path(f'checkpoints/{name}')
  if path.exists() == False:
    path.mkdir(parents=True, exist_ok=True)
  torch.save(model.state_dict(), f'checkpoints/{name}/{name}_{epoch}.pth')


def denormalizeTensorImg(img:torch.Tensor, std:torch.Tensor = None, mean:torch.Tensor = None) -> torch.Tensor:
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  return img * std + mean


def saveFile(path:Path, obj):
  with open(path, 'wb') as file:
    pickle.dump(obj, file)


def loadFile(path:Path, obj):
  with open(path, 'rb') as file:
    return pickle.load(file)