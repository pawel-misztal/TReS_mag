from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd

BIQ2021_PATH = Path("/home/mrpaw/Documents/mag_databases/BIQ2021/BIQ2021-main")

class BIQ2021Dataset(Dataset):
  """
  mos - higher is better 0-1\n
  real distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.testDataPath = path / "Test (Images and MOS).csv"
    self.trainDataPath = path / "Train (Images and MOS).csv"
    self.imagesPath = path / "Images"

    self.dataPath = self.trainDataPath if train else self.testDataPath

    scores = pd.read_csv(self.dataPath)
    self.mos = scores["MOS"].values
    self.images = scores["Image Name"].values

    length = len(self.mos)

    self.indexes = np.arange(start=0, stop=length) 
    
    self.transform = transform

  def __len__(self):
    return len(self.indexes)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    i = self.indexes[index]
    img_path = self.imagesPath / self.images[i]
    mos = self.mos[i]

    img = Image.open(img_path)

    if(self.transform != None):
      img = self.transform(img)

    mos = torch.tensor(mos)

    return (img, mos)