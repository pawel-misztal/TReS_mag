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
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, seed=2137, testSize = 0.2, normalize = False, load_img = True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.dmosPath = path / "dmos.mat"
    self.dmosRealignedPath = path / "dmos_realigned.mat"
    self.refnames_all = path / "refnames_all.mat"
    self.normalize = normalize

    jp2kPaths = self.sortPaths(list((path / "jp2k").glob("*.bmp")))
    jpegPaths = self.sortPaths(list((path / "jpeg").glob("*.bmp")))
    wnPaths = self.sortPaths(list((path / "wn").glob("*.bmp")))
    gblurPaths = self.sortPaths(list((path / "gblur").glob("*.bmp")))
    fastfadingPaths = self.sortPaths(list((path / "fastfading").glob("*.bmp")))

    self.imgsPaths = jp2kPaths + jpegPaths + wnPaths + gblurPaths + fastfadingPaths

    self.dmos =  sio.loadmat(self.dmosPath)
    self.dmosRel = sio.loadmat(self.dmosRealignedPath)
    self.refnames_all = sio.loadmat(self.refnames_all)
    # print(self.dmos.keys())
    # print(self.dmosRel.keys())
    # print(self.refnames_all.keys())
    # print(self.dmosRel["dmos_new"][0][0:10])
    # print(self.dmosRel["orgs"][0][0:10])
    # print(self.refnames_all["refnames_all"][0][0:10])
    
    mask = self.dmosRel["orgs"][0]
    self.maskedDmos = self.dmosRel["dmos_new"][0][mask == 0]
    self.maskedNames = self.refnames_all["refnames_all"][0][mask == 0]
    self.maskedPaths = np.array(self.imgsPaths)[mask == 0]
    # print("----------------------------")
    # print(self.maskedDmos[0:10])
    # print(self.maskedNames[0:10])
    # print(self.maskedPaths[0:10])
    # print(len(self.maskedDmos))

    length = len(self.maskedDmos)
    
    assert len(self.maskedPaths) == length, "lenths should be the same"

    self.indexes = np.arange(start=0, stop=length) 

    
    unique_refs = np.unique(self.maskedNames)
    # print(unique_refs)

    # i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=21, shuffle=True)
    ref_train, ref_test = train_test_split(unique_refs, test_size=testSize, random_state=seed, shuffle=True)
    # print(len(self.maskedNames))
    # print(np.isin(self.maskedNames, ref_train)[0:10])
    # print(self.maskedNames[0:10])
    if train:
        selected_mask = np.isin(self.maskedNames, ref_train)
    else:
        selected_mask = np.isin(self.maskedNames, ref_test)

    # self.indexes = i_train if train else i_test
    self.indexes = self.indexes[selected_mask]
    
    self.transform = transform
    self.load_img = load_img

  def __len__(self):
    return len(self.indexes)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    i = self.indexes[index]
    img_path = self.maskedPaths[i] 
    dmos = self.maskedDmos[i]

    # print(img_path)
    if self.load_img:
      img = Image.open(img_path)

      if(self.transform != None):
        img = self.transform(img)
    else:
      img = None


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
     

if __name__ == "__main__":
  train = LIVEDataset(LIVE_PATH, True, None, normalize=False,load_img=False)
  test = LIVEDataset(LIVE_PATH, False, None, normalize=False,load_img=False)

  print("Test", len(test))
  print("Train", len(train))
  print("sum", len(test) + len(train))
