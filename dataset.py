import torch
import os
import pandas as pd
import math
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as T
import numpy as np

class PascalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, num_classes, fold_indexes, transform = None):
        self.annotations = pd.read_csv(csv_file,names=["images","labels"])
        self.annotations = self.annotations.iloc[fold_indexes] # slice data according to the fold
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.num_classes = num_classes
        self.fold_indexes = fold_indexes
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_dir = os.path.join(self.img_dir,self.annotations.iloc[index,0])
        label_dir = os.path.join(self.label_dir, self.annotations.iloc[index,1])
        classes = []

        with open(label_dir) as f:
            for row in f.readlines():
                elems = row.split()
                classes.append(elems[0])
    

        image = Image.open(image_dir)
        min_index = np.argmin(image.size)
        min_value = image.size[min_index]
        transform = T.Compose([
            OneDimResize(256,min_index),
            T.CenterCrop(227),
            T.ToTensor()
        ])
        image = transform(image)


        labels = torch.zeros(20)
        for c in classes:

            if labels[int(c)] == 0:
                labels[int(c)] = 1
        
        return image, labels


class OneDimResize:
    def __init__(self,size, min_index):
        self.size = size
        self.min_index = min_index
        

    def __call__(self,img):
        w, h = img.size
        aspect_ratio = float(h) / float(w)
        new_dim = int(self.size * aspect_ratio)
        resize_size = (self.size,new_dim) if self.min_index == 0 else (new_dim,self.size)
        return F.resize(img,size=resize_size)


