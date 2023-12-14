from torch.utils.data import Dataset
import os
import torch.nn.functional as F

class VAEDataset(Dataset):
    def __init__(self, 
                 part, 
                 data_path, 
                 limit=None):
        self.part = part
        self.image_path_folder = data_path
        self.limit = limit

        self.images = os.listdir(self.image_path_folder)
        self.images = list(map(lambda x: self.image_path_folder + "/" + x, self.images))

        if limit is not None:
            self.images = self.images[:limit]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return {"image" : self.images[index]}