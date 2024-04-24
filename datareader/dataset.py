from torch.utils.data import Dataset
import pathlib
from PIL import Image
from torch import Tensor, tensor
import torch
import numpy as np
from torchvision import transforms as T
from typing import Optional

class GameScreenShotDataset(Dataset):
    def __init__(self, root, transform: Optional[T.Compose] =None):
        self.root = root
        self.transform = transform
        
        self.classes = {}
        self.n = 0
        self.data_folder = []
        self.data_path = []
        
        # get all classes
        for folder in pathlib.Path(self.root).iterdir():
            folder_id = int(folder.name.split(" ")[0])
            folder_name = folder.name.split(" ")[1:]
            
            self.classes[folder_id] = " ".join(folder_name)
            
            self.data_path.extend(list(folder.iterdir()))
            
        # print(f"Classes: {self.classes}")
        # Sort the classes
        # i know this look kinda stupid, but if it works don't fix it
        self.classes = list(dict(sorted(self.classes.items())).values())
        
        # print(f"Sorted Classes: {self.classes}")
        
    def __len__(self):
        return len(self.data_path)
        # return len(self.data_folder)
        
    def __getitem__(self, idx) -> tuple[Tensor,Tensor]:
        img_path = self.data_path[idx]
        img = Image.open(img_path).convert("HSV")
        # convert to torch
        if self.transform:
            img_tensor: Tensor = self.transform(img) # type: ignore
        else:
            img_tensor = T.ToTensor()(img)
            
        class_id = int(img_path.parent.name.split(" ")[0])
        # y = np.zeros(len(self.classes))
        # y[class_id] = 1
        
        return img_tensor, tensor(class_id).long()