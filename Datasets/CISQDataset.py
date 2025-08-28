from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image

CISQ_PATH = Path("/home/mrpaw/Documents/mag_databases/CISQ")

class CISQDataset(Dataset):
  """
  dmos - lower is better 0-1\n
  artificial distortions
  """


  csvToFolder = {
    "noise": "awgn",
    "jpeg": "jpeg",
    "jpeg 2000": "jpeg2000",
    "fnoise" : "fnoise",
    "blur": "blur",
    "contrast": "contrast"
  }

  csvToFile = {
    "noise": "AWGN",
    "jpeg": "JPEG",
    "jpeg 2000": "jpeg2000",
    "fnoise" : "fnoise",
    "blur": "BLUR",
    "contrast": "contrast"
  }

  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, seed = 21, load_img = True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.dataPath = path / "csiq.DMOS.fixed.csv"
    self.imgsPaths = path / "dst_imgs"

    self.data =  pd.read_csv(self.dataPath)
    length = len(self.data["dmos"])


    names =  [n for n in self.data["image"]]
    unitque_names =np.unique(names)
    n_train, n_test = train_test_split(unitque_names, test_size=testSize, random_state=seed, shuffle=True)
    # print(n_train)
    # print("names",names)
    mask = np.isin(names, n_train) if train else np.isin(names, n_test) 


    self.indexes = np.arange(start=0, stop=length) 
    self.indexes = self.indexes[mask]

    # i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=seed, shuffle=True)

    # self.indexes = i_train if train else i_test
    
    self.transform = transform
    self.load_img = load_img

  def __len__(self):
    return len(self.indexes)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    i = self.indexes[index]
    distortType = self.data["dst_type"][i]
    distortAmount = self.data["dst_lev"][i]
    imgName = self.data["image"][i]
    dmos = self.data["dmos"][i]
    img_path = self.imgsPaths / self.csvToFolder[distortType] / f"{imgName}.{self.csvToFile[distortType]}.{distortAmount}.png"

    if self.load_img:
      img = Image.open(img_path)

      if(self.transform != None):
        img = self.transform(img)
    else: 
      img = None

    dmos = torch.tensor(dmos, dtype=torch.float32).unsqueeze(0)

    return (img, dmos)
     
if __name__ == "__main__":
  test_set = CISQDataset(CISQ_PATH, train=False,load_img=False)
