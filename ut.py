import unittest
import model as m
import dataset as d
import train as t
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import utils
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import os
from PIL import Image
"""
Contains unit test for source codes.
"""
class Tester(unittest.TestCase):
    def __init__(self):
        super(Tester,self).__init__()
        self.annotations = pd.read_csv("../data/PascalVOC/train.csv",names=["images","labels"])

    def test_dataset(self,train_index, valid_index):

        def visualize():
            for i in range(4):
                
                idx = np.random.choice(train_index)
                image  = np.transpose(train_dataset[idx][0].numpy(),(1,2,0))
                
                cl = [idx for idx, _ in enumerate(train_dataset[idx][1].numpy()) if _ == 1]
                title = [t.classes[label] for label in cl]
                plt.subplot(2,2,i+1) 
                plt.title(title)
                plt.imshow(image)
            plt.show()

        kf = KFold(n_splits=4)
        print("Shape of annotations before slicing: {} \nDivided by batch size of {}: {}".format(self.annotations.shape, t.BATCH_SIZE ,self.annotations.shape[0] / t.BATCH_SIZE))
        
        train_dataset = d.PascalDataset(t.TRAIN_CSV, t.IMG_DIR,t.LABEL_DIR, t.NUM_CLASSES, train_index)
        valid_dataset = d.PascalDataset(t.TRAIN_CSV, t.IMG_DIR,t.LABEL_DIR, t.NUM_CLASSES, valid_index)
        
        train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= t.BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers= 8,
        drop_last=True
        )
        valid_loader = DataLoader(
            dataset= valid_dataset,
            batch_size= t.BATCH_SIZE,
            pin_memory=True,
            shuffle=True,
            num_workers= 8,
            drop_last=True
        )

        for i,_ in enumerate(train_index):
            assert train_dataset[i][0].shape == torch.Size([3,t.IMG_SIZE,t.IMG_SIZE]), f"Input shape should be (3, {t.IMG_SIZE}, {t.IMG_SIZE}) as in (channels, height, width)"
            assert train_dataset[i][1].shape == torch.Size([t.NUM_CLASSES]), f"Label shape must be {t.NUM_CLASSES} in training set"

        for i,_ in enumerate(valid_index):
            assert valid_dataset[i][0].shape == torch.Size([3,t.IMG_SIZE,t.IMG_SIZE]), f"Input shape should be (3, {t.IMG_SIZE}, {t.IMG_SIZE}) as in (channels, height, width)"
            assert valid_dataset[i][1].shape == torch.Size([t.NUM_CLASSES]), f"Label shape must be {t.NUM_CLASSES} in validation set"

        visualize()
        

    def test_model(self):
        model = m.AlexNet(t.NUM_CLASSES)
        input = torch.randn(1,3,227,227)
        output = model(input)
        print(output.shape)


    def test_class_distribution(self,dataset):
        utils.plot_class_dist(dataset)



def main():
    tester = Tester()
    # train_index = np.random.randint(0,1000,size=(1000,))
    # valid_index = [3,4,5]
    # tester.test_dataset(train_index, valid_index)
    tester.test_model()
if __name__ == "__main__":
    main()


