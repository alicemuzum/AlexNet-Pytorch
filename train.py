import torch
from model import AlexNet
import dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import os
#from tensorboardX import SummaryWriter


TRAIN_CSV = "../data/PascalVOC/train.csv"
TEST_CSV = "../data/PascalVOC/train.csv"
IMG_DIR = "../data/PascalVOC/images"
LABEL_DIR = "../data/PascalVOC/labels"
LOG_DIR = "log"
CHECKPOINT_DIR = "models"
NUM_CLASSES = 20
BATCH_SIZE = 128
NUM_EPOCHS = 90
MOMENTUM = 0.9
W_DECAY = 0.0005
W_INIT = 0.01
LR = 0.01

"""
To Do:
    - Data Augmentation kısmını impleme et.
    + Weight Inıt
    - AlexNEtteki kwargsa bak
    + Add validataion set
    - Use tensorboardX
    - * ne işe yarrıyor not al
    - Test csvden sonunc al
    - loss foksiyonunu kontrol et çünkü outputa uygun olmayabilir
"""

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

classes = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',7:'cat',8:'chair',
               9:'cow',10:'diningtable',11:'dog',12:'horse',13:'motorbike',14:'person',15:'potted_plant',
               16:'sheep',17:'sofa',18:'train',19:'tv_monitor'}

def train(model,device,train_loader,optimizer):
    loop = tqdm(train_loader)
    mean_loss = []
    train_loss,train_correct=0.0,0
    model.train()
    for img_batch, label_batch in loop: # 128 adet imagedan oluşan tüm batchleri teker teker döndürüyor. imgsda 128 adet img var
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        output = model(img_batch)
        loss = F.cross_entropy(output,label_batch)
        mean_loss.append(loss.item())

        optimizer.zero_grad()           # zero out the gradients because pytorch accumulates them. we don't want to use previous gradients
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())
        train_loss += loss.item() * img_batch.size(0)
        scores, predictions = torch.max(output.data, 1)
        scores,labels = torch.max(label_batch.data,1)
        train_correct += (predictions == labels).sum().item()
    
    return train_loss, train_correct


def valid(model,device,valid_loader):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss = F.cross_entropy(output,labels)
            valid_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            scores, l_argmax = torch.max(labels.data,1)
            val_correct+=(predictions == l_argmax).sum().item()

    return valid_loss,val_correct


def main():
    seed = torch.initial_seed()
    model = AlexNet(NUM_CLASSES).to(device)
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    optimizer = optim.Adam(
            params=model.parameters(),
            lr=LR,
            #momentum=MOMENTUM,
            weight_decay=W_DECAY)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    data = pd.read_csv(TRAIN_CSV,names=["images", "labels"])
    total_steps = 0
    k = 8
    kf = KFold(n_splits=k)

    for fold, (train_index, valid_index) in enumerate(kf.split(data["images"], data["labels"])):
        
        train_dataset = dataset.PascalDataset(TRAIN_CSV, IMG_DIR,LABEL_DIR, NUM_CLASSES,train_index)
        valid_dataset = dataset.PascalDataset(TRAIN_CSV, IMG_DIR,LABEL_DIR, NUM_CLASSES, valid_index)
        
        train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers= 8,
        drop_last=True
        )
        valid_loader = DataLoader(
            dataset= valid_dataset,
            batch_size= BATCH_SIZE,
            pin_memory=True,
            shuffle=True,
            num_workers= 8,
            drop_last=True
        )

        for epoch in range(NUM_EPOCHS):
            train_loss, train_correct = train(model,device,train_loader,optimizer)
            valid_loss, test_correct=valid(model,device,valid_loader)
            
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            valid_loss = valid_loss / len(valid_loader.sampler)
            test_acc = test_correct / len(valid_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f} AVG Training Acc {:.2f} % AVG Valid Acc {:.2f} %".format(epoch + 1,
                                                                                                             NUM_EPOCHS,
                                                                                                             train_loss,
                                                                                                             valid_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(valid_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)   

    avg_train_loss = np.mean(history['train_loss'])
    avg_test_loss = np.mean(history['test_loss'])
    avg_train_acc = np.mean(history['train_acc'])
    avg_test_acc = np.mean(history['test_acc'])
    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}"
          .format(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc))  

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
    state = {
        'epoch': epoch,
        'total_steps': total_steps,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'seed': seed,
    }
    torch.save(state, checkpoint_path)

if __name__ == "__main__":
    main()
    
