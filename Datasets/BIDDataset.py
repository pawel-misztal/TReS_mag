from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.io as sio

BID_PATH = Path("/home/mrpaw/Documents/mag_databases/BID/BID-20250703T072317Z-1-001/BID")

class BIDDataset(Dataset):
  """
  mos - higher is better 0-5 -> 0-1\n
  real distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, normalize = False, seed=2137,loadImg=True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.normalize = normalize
    self.imagesPath = path / "ImageDatabase"

    imdb = sio.loadmat(self.path / "imdb.mat")
    # print(imdb.keys())
    # print(len(imdb["images"][0][0]))
    # print(imdb["images"][0][0][0][0:10])
    # print(imdb["images"][0][0][1][0][0:10])
    # print(len(imdb["path"][0]))

    self.mos = imdb["images"][0][0][0]
    self.images = imdb["images"][0][0][1][0]

    self.indexes = np.arange(start=0, stop=len(self.mos)) 
    i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=seed, shuffle=True)

    self.indexes = i_train if train else i_test
    
    self.transform = transform
    self.load_img = loadImg

  def __len__(self):
    return len(self.indexes)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    i = self.indexes[index]
    img_path = self.imagesPath / self.images[i]
    mos = self.mos[i]

    # print(type(img_path[0]))
    # print(img_path)
    if self.load_img:
      img = Image.open(img_path[0])

      if(self.transform != None):
        img = self.transform(img)
    else:
      img = None

    if(self.normalize):
      mos = mos / 5.0

    mos = torch.tensor(mos, dtype=torch.float32)

    return (img, mos)
  

if __name__ == "__main__":
    dset = BIDDataset(BID_PATH, True, loadImg=True)

    i = 0
    for d in dset:
        i += 1
        print("mos",d[1])
        if(i > 10):
            break