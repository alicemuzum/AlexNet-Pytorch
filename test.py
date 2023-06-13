import torch 
import dataset 
import model
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import numpy as np
import utils
import pickle

TEST_CSV = "../data/PascalVOC/test.csv"
IMG_DIR = "../data/PascalVOC/images"
LABEL_DIR = "../data/PascalVOC/labels"
BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
def test():
    # with open('models/overfit.pkl', 'rb') as f:
    #     models = pickle.load(f) # deserialize using load()
    #     print(models)
    # model.load_state_dict(torch.load(PATH))
    indexes = np.arange(1,4952)
    test_dataset = dataset.PascalDataset(TEST_CSV,IMG_DIR,LABEL_DIR,20,indexes)
    test_loader = DataLoader(
        test_dataset,
        batch_size= BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        )
    net = model.AlexNet(20).to(device)
    net.load_state_dict(torch.load("models/wdecay-0.00005_epoch-120")['model'])
    net.eval()
    mean_loss = []
    test_metrics = {'acc':[], 'hamming_loss':[], 'precision':[], 'recall':[], 'f1':[]}
    with torch.no_grad():
        for images, label_batch in test_loader:
            images, label_batch = images.to(device), label_batch.to(device)
            output = net(images)
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
                test_metrics[m].append(metrics[m])
    
    for i in test_metrics:
        test_metrics[i] = sum(test_metrics[i]) / len(test_metrics[i])

    loss = sum(mean_loss) / BATCH_SIZE
    print(loss, test_metrics)

if __name__ == "__main__":
    test()