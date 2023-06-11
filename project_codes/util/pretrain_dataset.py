# %%
import os

import numpy as np
from random import sample
from PIL import Image
import copy

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision


class Pretrain_Dataset(data.Dataset):
    NUM_CLASS = 2
    def __init__(self, root='/home/tnt/Downloads/NIH_images', split='train', transform=None, args=None):
        self.root = root
        self.img_names_list = self._get_names()
        
        self.transform = transform

    def __getitem__(self, idx):
        img_path  = os.path.join(self.root, self.img_names_list[idx])
        img = Image.open(img_path)
        print(self.img_names_list[idx])
        
        if self.transform is not None:
            img = self.transform(img)
        img_np = np.array(img)
        img_np = np.expand_dims(img_np, axis=2)
        img_ten = self._img_transform(img_np)
        print(img_ten.shape)
        
        target = copy.deepcopy(img_ten)

        return img_ten, target

    def _get_names(self):
        img_names_list = os.listdir(self.root)
        return img_names_list
    
    def _img_transform(self, img):
        #img = np.ascontiguousarray(img.transpose(2, 0, 1))
        #return torchvision.transforms.functional.to_tensor(img)
        return torch.from_numpy(img)
    
    def __len__(self):
        return len(self.img_names_list)






# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
