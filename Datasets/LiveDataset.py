from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os, re

LIVE_PATH = Path("/home/mrpaw/Documents/mag_databases/LIVE/databaserelease2")


#TODO usuanąć refimage ze zbioru, podzielić zdjęcia po bazowym zdjęciu
class LIVEDataset(Dataset):
  """
  dmos - lower is better 0-100 -> 0-1\n
  artificial distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, normalize = True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.dmosPath = path / "dmos.mat"
    self.normalize = normalize

    jp2kPaths = self.sortPaths(list((path / "jp2k").glob("*.bmp")))
    jpegPaths = self.sortPaths(list((path / "jpeg").glob("*.bmp")))
    wnPaths = self.sortPaths(list((path / "wn").glob("*.bmp")))
    gblurPaths = self.sortPaths(list((path / "gblur").glob("*.bmp")))
    fastfadingPaths = self.sortPaths(list((path / "fastfading").glob("*.bmp")))

    self.imgsPaths = jp2kPaths + jpegPaths + wnPaths + gblurPaths + fastfadingPaths

    self.dmos =  sio.loadmat(self.dmosPath)
    length = len(self.dmos["dmos"][0])
    
    assert len(self.imgsPaths) == length, "lenths should be the same"

    self.indexes = np.arange(start=0, stop=length) 

    i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=21, shuffle=True)

    self.indexes = i_train if train else i_test
    
    self.transform = transform

  def __len__(self):
    return len(self.indexes)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    i = self.indexes[index]
    img_path = self.imgsPaths[i] 
    dmos = self.dmos["dmos"][0][i]

    # print(img_path)
    img = Image.open(img_path)

    if(self.transform != None):
      img = self.transform(img)

    if(self.normalize):
      dmos = dmos / 100

    dmos = torch.tensor(dmos)


    return (img, dmos)
  
  def sortFunc(self,path):
    filename = os.path.basename(path)
    m = re.match(r"img(\d+)\.bmp", filename)
    if m:
        return int(m.group(1))
    else:
        return float('inf')
    
  def sortPaths(self,paths): 
    return  sorted(paths,key=self.sortFunc)
     
     