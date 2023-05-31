import torch
from model import AlexNet
import dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
#from tensorboardX import SummaryWriter


TRAIN_CSV = "../data/PascalVOC/train.csv"
TEST_CSV = "../data/PascalVOC/train.csv"
IMG_DIR = "../data/PascalVOC/images"
LABEL_DIR = "../data/PascalVOC/labels"
LOG_DIR = "/log"
CHECKPOINT_DIR = "/models"
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
    - Add validataion set
    - Use tensorboardX
    - * ne işe yarrıyor not al
"""

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

classes = {0:'aeroplane',1:'bicycle',2:'bird',3:'boat',4:'bottle',5:'bus',6:'car',7:'cat',8:'chair',
               9:'cow',10:'diningtable',11:'dog',12:'horse',13:'motorbike',14:'person',15:'potted_plant',
               16:'sheep',17:'sofa',18:'train',19:'tv_monitor'}

def train_fn(train_loader, model, optimizer, loss_fn):
    """
        Takes DataLoader, Model, Optimizer, Loss Function
        Does forward and backward propagation
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop): # x -> (16, 3, 448,448)  y -> (16,7,7,30)
        x ,y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def main():
    seed = torch.initial_seed()
    #tbwriter = SummaryWriter(log_dir=LOG_DIR)
    model = AlexNet(NUM_CLASSES).to(device)

    optimizer = optim.SGD(
            params=model.parameters(),
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=W_DECAY)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_dataset = dataset.PascalDataset(TRAIN_CSV, IMG_DIR,LABEL_DIR, NUM_CLASSES)
    test_dataset = dataset.PascalDataset(TEST_CSV, IMG_DIR,LABEL_DIR, NUM_CLASSES)

    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers= 8,
        drop_last=True

    )
    test_loader = DataLoader(
        dataset= test_dataset,
        batch_size= BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers= 8,
        drop_last=True

    )

    total_steps = 0
    loop = tqdm(train_loader, leave=True)

    for epoch in range(NUM_EPOCHS):
        mean_loss = []
        for imgs, classes in loop:
            imgs, classes = imgs.to(device), classes.to(device)

            output = model(imgs)
            loss = F.cross_entropy(output,classes)
            mean_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            loop.set_postfix(loss = loss.item())
        
        
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