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
import utils 
import numpy as np
import os
import logging
import warnings
import json



warnings.filterwarnings("always")
# from tensorboardX import SummaryWriter

IMG_SIZE = 227
TRAIN_CSV = "../data/PascalVOC/train.csv"
TEST_CSV = "../data/PascalVOC/train.csv"
IMG_DIR = "../data/PascalVOC/images"
LABEL_DIR = "../data/PascalVOC/labels"
LOG_DIR = "log"
CHECKPOINT_DIR = "models"
OUTPUT_FILENAME = "wdecay-0.00005_epoch-120"
NUM_CLASSES = 20
BATCH_SIZE = 64
NUM_EPOCHS = 60
MOMENTUM = 0.9
W_DECAY = 0.00005
W_INIT = 0.01
LR = 0.0001
K = 2
SCHEDULER_STEP = 45
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "potted_plant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tv_monitor",
}
"""
To Do:
    - Data Augmentation kısmını impleme et.
    + Weight Inıt
    - AlexNEtteki kwargsa bak
    + Add validataion set
    - Use tensorboardX
    - * ne işe yarrıyor not al
    + Test csvden sonunc al
    + loss foksiyonunu kontrol et çünkü outputa uygun olmayabilir
    + Accuracy neden sıfırda kalıyor bul.
    - Overfiti halllet
    - Precision recall f1 nedir anla
    - Class dist plot
"""


def train(model, device, train_loader, optimizer):
    loop = tqdm(train_loader)
    mean_loss = []
    train_metrics = {'acc':[], 'hamming_loss':[], 'precision':[], 'recall':[], 'f1':[]}
    model.train()
    for img_batch, label_batch in loop:
        # send one batch of data  to gpu if available
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        # training
        optimizer.zero_grad()
        output = model(img_batch)
        target = label_batch.float()
        loss = F.binary_cross_entropy_with_logits(output, target)
        mean_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

        # necesseray for sklearn metrics
        output = F.sigmoid(output)
        label_batch = label_batch.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        y_pred = []
        for sample in output:
            y_pred.append([1 if i >= 0.5 else 0 for i in sample])
        y_pred = np.array(y_pred)

        metrics = utils.get_metrics(label_batch,y_pred)
        for m in metrics:
            train_metrics[m].append(metrics[m])

    for i in train_metrics:
        train_metrics[i] = sum(train_metrics[i]) / len(train_metrics[i])
    loss = sum(mean_loss) / BATCH_SIZE
    return loss, train_metrics
            
    
    

def valid(model, device, valid_loader):
    mean_loss = []
    valid_metrics = {'acc':[], 'hamming_loss':[], 'precision':[], 'recall':[], 'f1':[]}
    # block layers like normalization or dropout for inference
    model.eval()
    with torch.no_grad():
        for images, label_batch in valid_loader:
            images, label_batch = images.to(device), label_batch.to(device)
            output = model(images)
            target = label_batch.float()
            loss = F.binary_cross_entropy_with_logits(output, target)
            mean_loss.append(loss.item())

            output = F.sigmoid(output)
            label_batch = label_batch.detach().cpu().numpy()
            output = output.detach().cpu().numpy()

            y_pred = []
            for sample in output:
                y_pred.append([1 if i >= 0.5 else 0 for i in sample])
            y_pred = np.array(y_pred)

            metrics = utils.get_metrics(label_batch,y_pred)
            for m in metrics:
                valid_metrics[m].append(metrics[m])
    
    for i in valid_metrics:
        valid_metrics[i] = sum(valid_metrics[i]) / len(valid_metrics[i])

    loss = sum(mean_loss) / BATCH_SIZE
    return  loss, valid_metrics
            

    


def main():
    # set a seed to make code reproducive
    torch.manual_seed(0)
    model = AlexNet(NUM_CLASSES).to(device)
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
        "train_hamming": [],
        "valid_hamming": [],
        "train_precision": [],
        "valid_precision": [],
        "train_recall": [],
        "valid_recall": [],
        "train_f1": [],
        "valid_f1": [],
    }
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=LR,
        weight_decay=W_DECAY,
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=0.1
    )
    data = pd.read_csv(TRAIN_CSV, names=["images", "labels"])
    kf = KFold(n_splits=K)

    # # one loop for cross validation, one loop for epochs, one loop for training batches
    # for fold, (train_index, valid_index) in enumerate(
    #     kf.split(data["images"], data["labels"])
    # ):
    print("opt lr:", optimizer.param_groups[0]["lr"])
    train_index = np.arange(0,int((data.shape[0] * 2) / 3))
    valid_index = np.arange(int((data.shape[0] * 2) / 3), data.shape[0])
    # create datasets with fold indexes
    train_dataset = dataset.PascalDataset(
        TRAIN_CSV, IMG_DIR, LABEL_DIR, NUM_CLASSES, train_index
    )
    valid_dataset = dataset.PascalDataset(
        TRAIN_CSV, IMG_DIR, LABEL_DIR, NUM_CLASSES, valid_index
    )

    #utils.plot_class_dist(train_dataset)
    #utils.plot_class_dist(valid_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    for epoch in range(NUM_EPOCHS):
        train_loss, train_metrics = train(model, device, train_loader, optimizer)
        valid_loss, validation_metrics = valid(model, device, valid_loader)

        lr_scheduler.step()

        print(
            "Epoch:{}/{} Training Loss:{:.3f} Valid Loss:{:.3f} Train Acc {:.2f} % Valid Acc {:.2f} % Hamming Loss {:.2f} Precision {:.2f} Recall {:.2f} F_1 {:.2f}".format(
                epoch + 1,
                NUM_EPOCHS,
                train_loss,
                valid_loss,
                train_metrics["acc"],
                validation_metrics["acc"],
                train_metrics["hamming_loss"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
            )
        )
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_metrics["acc"])
        history["valid_acc"].append(validation_metrics["acc"])
        history["train_hamming"].append(train_metrics["hamming_loss"])
        history["valid_hamming"].append(validation_metrics["hamming_loss"])
        history["train_precision"].append(train_metrics["precision"])
        history["valid_precision"].append(validation_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["valid_recall"].append(validation_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["valid_f1"].append(validation_metrics["f1"])

    history["RUN"] = 0
    with open("log/history.json", "a") as f:
        json.dump(history, f)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, OUTPUT_FILENAME
    )
    state = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(),
    }
    torch.save(state, checkpoint_path)

    plt.subplot(2,1,1) 
    plt.title("Loss")
    plt.plot(range(K * NUM_EPOCHS), history['train_loss'], "r", range(K * NUM_EPOCHS), history['valid_loss'],"g")

    plt.subplot(2,1,2) 
    plt.title("Accuracy")
    plt.plot(range(K * NUM_EPOCHS), history['train_acc'], "r", range(K * NUM_EPOCHS), history['valid_acc'],"g")

    plt.show()

    avg_train_loss = np.mean(history["train_loss"])
    avg_test_loss = np.mean(history["valid_loss"])
    avg_train_acc = np.mean(history["train_acc"])
    avg_test_acc = np.mean(history["valid_acc"])
    avg_train_hamming = np.mean(history["train_hamming"])
    avg_valid_hamming = np.mean(history["valid_hamming"])
    avg_train_precision = np.mean(history["train_precision"])
    avg_valid_precision = np.mean(history["valid_precision"])
    avg_train_recall = np.mean(history["train_recall"])
    avg_valid_recall = np.mean(history["valid_recall"])
    avg_train_f1 = np.mean(history["train_f1"])
    avg_valid_f1 = np.mean(history["valid_f1"])

    print("Performance of {} fold cross validation".format(K))
    logging.basicConfig(
        filename="history.log", format="%(name)s - %(levelname)s - %(message)s", level= logging.INFO
    )
    logging.info(
        "Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}\n Avg Train Hamming {:.2f} \t Avg Valid Hamming {:.2f} \t Avg Train Precision {:.2f} \t Avg Valid Precision {:.2f} \t Avg Train Recall {:.2f} \t Avg Valid Recall {:.2f} \t Avg Train F1 {:.2f} \t Avg Valid F1 {:.2f} ".format(
            avg_train_loss,
            avg_test_loss,
            avg_train_acc,
            avg_test_acc,
            avg_train_hamming,
            avg_valid_hamming,
            avg_train_precision,
            avg_valid_precision,
            avg_train_recall,
            avg_valid_recall,
            avg_train_f1,
            avg_valid_f1,
        )
    )


if __name__ == "__main__":
    main()
