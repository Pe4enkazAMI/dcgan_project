from torch.utils.data import Dataset
import os
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import glob

class VAEDataset(Dataset):
    def __init__(self, 
                 part, 
                 data_path, 
                 limit=None):
        self.part = part
        self.image_path_folder = data_path
        self.limit = limit
        self.transforms = T.ToTensor()

        self.images = glob.glob(f"{self.image_path_folder}/*.png")

        if limit is not None:
            self.images = self.images[:limit]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        img = Image.open(img).convert("RGB")
        return {"image" : self.transforms(img)}