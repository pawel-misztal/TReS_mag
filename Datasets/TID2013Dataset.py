from torch.utils.data import Dataset
import torch;
from typing import Tuple
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


TID2013_PATH  = Path("/home/mrpaw/Documents/mag_databases/tid2013")

#TODO podzielić zdjęcia po bazowym zdjęciu
class TID2013Dataset(Dataset):
  """
  mos - higher is better [0-8) -> 0-1\n
  artificial distortions
  """
  def __init__(self, path:Path, train:bool, transform:torch.nn.Module = None, testSize = 0.2, normalize = False, seed = 21, load_img = True) -> None:
    super().__init__()
    self.train = train
    self.path = path
    self.dataPath = path / "mos_with_names.txt"
    self.imagesPath = path / "distorted_images"
    self.normalize = normalize

    scores = pd.read_csv(self.dataPath, sep=" ",header=None, names=['mos', 'name'], encoding='utf-8')
    self.mos = scores["mos"].values
    self.images = scores["name"].values

    ref_ids = [name.split('_')[0] for name in self.images]  # e.g. 'i01_00_1.png' -> 'i01'
    ref_ids = np.array(ref_ids)
    unique_refs = np.unique(ref_ids)

    ref_train, ref_test = train_test_split(unique_refs, test_size=testSize, random_state=seed, shuffle=True)

    if train:
        selected_mask = np.isin(ref_ids, ref_train)
    else:
        selected_mask = np.isin(ref_ids, ref_test)

    # length = len(self.mos)
    # self.indexes = np.arange(start=0, stop=length) 
    # i_train, i_test = train_test_split(self.indexes, test_size=testSize, random_state=21, shuffle=True)
    # self.indexes = i_train if train else i_test
    

    self.images = self.images[selected_mask]
    self.mos = self.mos[selected_mask]


    self.transform = transform
    self.load_img = load_img

  def __len__(self):
    return len(self.images)  

  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    # i = self.indexes[index]
    img_path = self.imagesPath / self.images[index]
    mos = self.mos[index]

    # print(img_path)
    if(self.load_img):
      img = Image.open(img_path)

      if(self.transform != None):
        img = self.transform(img)
    else:
      img = None

    if(self.normalize):
      mos = mos / 8.0

    mos = torch.tensor(mos, dtype=torch.float32).unsqueeze(0)

    return (img, mos)
    

if __name__ == "__main__":
  # import matplotlib.pyplot as plt
  dset = TID2013Dataset(TID2013_PATH, True, None, load_img=False)
  img, mos = dset[3]
  # print(mos)
  # plt.imshow(img)
  # plt.show()

  # for i, (_, m) in enumerate(dset):
  #   if i > 20:
  #     break
  