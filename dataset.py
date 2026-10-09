import torch
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as T

class PascalDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, num_classes, fold_indexes, train=False):
        super(PascalDataset,self).__init__()
        self.annotations = pd.read_csv(csv_file,names=["images","labels"])  
        self.annotations = self.annotations.iloc[fold_indexes] # slice data according to the fold
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.num_classes = num_classes
        self.fold_indexes = fold_indexes
        self.train = train
    
    def get_annotations(self):
        return self.annotations
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_dir = os.path.join(self.img_dir,self.annotations.iloc[index,0])
        label_dir = os.path.join(self.label_dir, self.annotations.iloc[index,1])
        classes = []

        with open(label_dir) as f:
            for l in f.readlines():
                classes.append(l.split()[0])
    

        image = Image.open(image_dir)
        if self.train:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(227),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(227),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        image = transform(image)


        labels = torch.zeros(self.num_classes)
        for c in classes:

            if labels[int(c)] == 0:
                labels[int(c)] = 1
        
        return image, labels



