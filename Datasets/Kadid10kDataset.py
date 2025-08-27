from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

KADID10K_PATH = Path("/home/mrpaw/Documents/mag_databases/KADID-10k/kadid10k")

class Kadid10kDataset(Dataset):
  """
  dmos - higher is better 0-5 -> 0-1\n
  artificial distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, normalize = True, seed:int=21, loadImg=True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.dataPath = path / "dmos.csv"
    self.imagesPath = path / "images"
    self.normalize = normalize

    scores = pd.read_csv(self.dataPath)
    self.mos = scores["dmos"].values
    self.images = scores["dist_img"].values

    # length = len(self.mos)

    # self.indexes = np.arange(start=0, stop=length) 


    # i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=21, shuffle=True)

    # self.indexes = i_train if train else i_test
    
    # self.transform = transform

    # Grupujemy obrazy według identyfikatora referencyjnego (pierwsze dwa znaki po 'i')
    ref_ids = [name.split('_')[0] for name in self.images]  # e.g. 'i01_00_1.png' -> 'i01'
    ref_ids = np.array(ref_ids)

    # Znajdujemy unikalne referencje i dzielimy je na train/test
    unique_refs = np.unique(ref_ids)
    # print(unique_refs)
    ref_train, ref_test = train_test_split(unique_refs, test_size=testSize, random_state=seed, shuffle=True)
    # print(ref_test)
    # print(ref_train)

    # Tworzymy maskę do filtrowania pełnych danych
    if train:
        selected_mask = np.isin(ref_ids, ref_train)
    else:
        selected_mask = np.isin(ref_ids, ref_test)

    self.images = self.images[selected_mask]
    # print(self.images)
    self.mos = self.mos[selected_mask]
    self.transform = transform
    self.load_img = loadImg

  def __len__(self):
    return len(self.images)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    # i = self.indexes[index]
    # img_path = self.imagesPath / self.images[i]
    img_path = self.imagesPath / self.images[index]
    mos = self.mos[index]

    if self.load_img:
      # print(img_path)
      img = Image.open(img_path)

      if(self.transform != None):
        img = self.transform(img)
    else:
      img = None

    if(self.normalize):
      mos = mos / 5.0

    mos = torch.tensor(mos, dtype=torch.float32).unsqueeze(0)

    return (img, mos)
  

if __name__ == "__main__":
  testDataset = Kadid10kDataset(KADID10K_PATH, False, None, normalize= False, loadImg=False)
  i = 0
  for t in testDataset:
    i += 1
    print(t[1])

    if i > 10: 
      break