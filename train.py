import torch
from model import AlexNet
import dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics 
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import os
import logging
import warnings
warnings.filterwarnings('always')
#from tensorboardX import SummaryWriter

IMG_SIZE = 227
TRAIN_CSV = "../data/PascalVOC/train.csv"
TEST_CSV = "../data/PascalVOC/train.csv"
IMG_DIR = "../data/PascalVOC/images"
LABEL_DIR = "../data/PascalVOC/labels"
LOG_DIR = "log"
CHECKPOINT_DIR = "models"
NUM_CLASSES = 20
BATCH_SIZE = 32
NUM_EPOCHS = 100
MOMENTUM = 0.9
W_DECAY = 0.0005
W_INIT = 0.01
LR = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
classes = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',7:'cat',8:'chair',
               9:'cow',10:'diningtable',11:'dog',12:'horse',13:'motorbike',14:'person',15:'potted_plant',
               16:'sheep',17:'sofa',18:'train',19:'tv_monitor'}
"""
To Do:
    - Data Augmentation kısmını impleme et.
    + Weight Inıt
    - AlexNEtteki kwargsa bak
    + Add validataion set
    - Use tensorboardX
    - * ne işe yarrıyor not al
    - Test csvden sonunc al
    + loss foksiyonunu kontrol et çünkü outputa uygun olmayabilir
    - Accuracy neden sıfırda kalıyor bul.
"""

def train(model,device,train_loader,optimizer):
    loop = tqdm(train_loader)
    mean_loss = []
    train_loss=0.0
    model.train()
    for img_batch, label_batch in loop: # 128 adet imagedan oluşan tüm batchleri teker teker döndürüyor. imgsda 128 adet img var
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)
        #scores,labels = torch.max(label_batch.data,1)

        optimizer.zero_grad()
        output = model(img_batch)
        target = label_batch.float()
        loss = F.binary_cross_entropy_with_logits(output,target)
        mean_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss = loss.item())
        train_loss += loss.item() * img_batch.size(0)
        scores, predictions = torch.max(output.data, 1)
        
        output = F.sigmoid(output)
        label_batch = label_batch.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        y_pred=[]
        for sample in  output:
            y_pred.append([1 if i>=0.5 else 0 for i in sample ] )
        y_pred = np.array(y_pred)

        exact_match = metrics.accuracy_score(label_batch,y_pred,normalize=True)    # Ne kadar prediction tamı tamına aynısı
        hamming_loss = metrics.hamming_loss(label_batch, y_pred)                   # Error rate, 0 iyi 1 kötü
        precision = metrics.precision_score(label_batch, y_pred,average="samples",zero_division=1) # Ne kadar negative olanı positive tahmin etmiyor. 1 en iyi
        recall = metrics.recall_score(label_batch, y_pred,average="samples")       # Ne kadar positive olanlara positive demiş. 1 en iyi
        f_1 = metrics.f1_score(label_batch, y_pred,average="samples")              # precision ve recallın oranı 1 iyi 0 kötü
    
    return {'train_loss':train_loss, 'exact_match':exact_match, 
            'hamming_loss':hamming_loss, 'precision':precision, 'recall':recall, 'f_1':f_1}


def valid(model,device,valid_loader):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, label_batch in valid_loader:
            images,label_batch = images.to(device),label_batch.to(device)
            output = model(images)
            target = label_batch.float()
            loss = F.binary_cross_entropy_with_logits(output,target)
            valid_loss+=loss.item()*images.size(0)

            output = F.sigmoid(output)
            label_batch = label_batch.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            

            y_pred=[]
            for sample in  output:
                y_pred.append([1 if i>=0.5 else 0 for i in sample ] )
            y_pred = np.array(y_pred)
            
            exact_match = metrics.accuracy_score(label_batch,y_pred,normalize=True)    
            hamming_loss = metrics.hamming_loss(label_batch, y_pred)                   
            precision = metrics.precision_score(label_batch, y_pred,average="samples",zero_division=1) 
            recall = metrics.recall_score(label_batch, y_pred,average="samples")       
            f_1 = metrics.f1_score(label_batch, y_pred,average="samples")

    return {'valid_loss':valid_loss, 'exact_match':exact_match, 
            'hamming_loss':hamming_loss, 'precision':precision, 'recall':recall, 'f_1':f_1}


def main():
    torch.manual_seed(0)
    model = AlexNet(NUM_CLASSES).to(device)
    history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[],
                'train_hamming':[], 'valid_hamming':[], 'train_precision':[], 'valid_precision':[],
                'train_recall':[], 'valid_recall':[], 'train_f1':[], 'valid_f1':[]}
    optimizer = optim.Adam(
            params=model.parameters(),
            lr=LR,
            #momentum=MOMENTUM,
            #weight_decay=W_DECAY
            )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    data = pd.read_csv(TRAIN_CSV,names=["images", "labels"])
    total_steps = 0
    k = 4
    kf = KFold(n_splits=k)

    for fold, (train_index, valid_index) in enumerate(kf.split(data["images"], data["labels"])):
        print("opt lr:", optimizer.param_groups[0]['lr'])
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
            
            train_metrics = train(model,device,train_loader,optimizer)
            validation_metrics =valid(model,device,valid_loader)

            lr_scheduler.step()

            train_loss = train_metrics['train_loss'] / len(train_loader.sampler)
            valid_loss = validation_metrics['valid_loss'] / len(valid_loader.sampler)
            

            print("Epoch:{}/{} Training Loss:{:.3f} Valid Loss:{:.3f} Train Acc {:.2f} % Valid Acc {:.2f} % Hamming Loss {:.2f} Precision {:.2f} Recall {:.2f} F_1 {:.2f}".format(epoch + 1,
                                                                                                             NUM_EPOCHS,
                                                                                                             train_loss,
                                                                                                             valid_loss,
                                                                                                             train_metrics['exact_match'],
                                                                                                             validation_metrics['exact_match'],
                                                                                                             train_metrics['hamming_loss'],
                                                                                                             train_metrics['precision'],
                                                                                                             train_metrics['recall'],
                                                                                                             train_metrics['f_1']))
            history['train_loss'].append(train_metrics['train_loss'])
            history['valid_loss'].append(validation_metrics['valid_loss'])
            history['train_acc'].append(train_metrics['exact_match'])
            history['valid_acc'].append(validation_metrics['exact_match'])
            history['train_hamming'].append(train_metrics['hamming_loss'])
            history['valid_hamming'].append(validation_metrics['hamming_loss'])
            history['train_precision'].append(train_metrics['precision'])
            history['valid_precision'].append(validation_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            history['valid_recall'].append(validation_metrics['recall'])
            history['train_f1'].append(train_metrics['f_1'])
            history['valid_f1'].append(validation_metrics['f_1'])


    avg_train_loss = np.mean(history['train_loss'])
    avg_test_loss = np.mean(history['valid_loss'])
    avg_train_acc = np.mean(history['train_acc'])
    avg_test_acc = np.mean(history['valid_acc'])
    avg_train_hamming = np.mean(history['train_hamming'])
    avg_valid_hamming = np.mean(history['valid_hamming'])
    avg_train_precision = np.mean(history['train_precision'])
    avg_valid_precision = np.mean(history['valid_precision'])
    avg_train_recall = np.mean(history['train_recall'])
    avg_valid_recall = np.mean(history['valid_recall'])
    avg_train_f1 = np.mean(history['train_f1'])
    avg_valid_f1 = np.mean(history['valid_f1'])



    print('Performance of {} fold cross validation'.format(k))
    logging.basicConfig(filename='history.log', format='%(name)s - %(levelname)s - %(message)s')
    logging.info("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}\n Avg Train Hamming {:.2f} \t Avg Valid Hamming {:.2f} \t Avg Train Precision {:.2f} \t Avg Valid Precision {:.2f} \t Avg Train Recall {:.2f} \t Avg Valid Recall {:.2f} \t Avg Train F1 {:.2f} \t Avg Valid F1 {:.2f} "
          .format(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc,avg_train_hamming, avg_valid_hamming, avg_train_precision, avg_valid_precision, avg_train_recall, avg_valid_recall, avg_train_f1, avg_valid_f1)) 
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
    state = {
        'epoch': epoch,
        'total_steps': total_steps,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }
    torch.save(state, checkpoint_path)

if __name__ == "__main__":
    main()
    
