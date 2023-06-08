import torch.nn as nn
import dataset
import train as t
import torch
import torch.nn.functional as F
import numpy as np
import model
from torch.utils.data import DataLoader
import pandas as pd 
from sklearn.model_selection import KFold
from tqdm import tqdm

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def get_index(matrix):
    result = []
    for sample in matrix:
        temp=[]
        for idx, pred in enumerate(sample):
            if pred == 1:
                temp.append(idx)
        result.append(temp)

    return result

def get_acc(y, y_hat):
    total = 0
    for sample in range(len(y)):
        if y[sample] == y_hat[sample]:
            total += 1
    return total / len(y)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
train_dataset = dataset.PascalDataset(t.TRAIN_CSV, t.IMG_DIR, t.LABEL_DIR, t.NUM_CLASSES, np.arange(0,16))
valid_dataset = dataset.PascalDataset(t.TRAIN_CSV, t.IMG_DIR, t.LABEL_DIR, t.NUM_CLASSES, np.arange(8,16))

train_loader = DataLoader(
dataset= train_dataset,
batch_size= 16,
pin_memory=True,
shuffle=False,
num_workers= 8,
drop_last=True
)
valid_loader = DataLoader(
    dataset= valid_dataset,
    batch_size= 8,
    pin_memory=True,
    shuffle=True,
    num_workers= 8,
    drop_last=True
)
torch.manual_seed(0)
model = AlexNet(20).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
for epoch in range(101):
    loop = tqdm(train_loader)
    for x , y in loop:
        x , y = x.to(device), y.to(device)
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        if epoch % 5 == 0:
            out = F.sigmoid(out)
            y_pred=[]
            for sample in out:
                y_pred.append([1 if i>=0.5 else 0 for i in sample ] )
            y_pred = np.array(y_pred)
        

    lr_scheduler.step()

y_pred = get_index(y_pred)
y = get_index(y)

for i in range(len(y)):
    print(y_pred[i],y[i])

acc = get_acc(y,y_pred)
print(acc)



