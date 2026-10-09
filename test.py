import torch 
import dataset 
import model
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import numpy as np
import utils
import train as t
import pandas as pd
import os
import sys

TEST_CSV = "../data/PascalVOC/test.csv"
IMG_DIR = "../data/PascalVOC/images"
LABEL_DIR = "../data/PascalVOC/labels"
BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = os.path.join(t.CHECKPOINT_DIR, t.OUTPUT_FILENAME)
if len(sys.argv) > 1:
    CHECKPOINT = sys.argv[1]
def test():
    data = pd.read_csv(TEST_CSV, names=["images", "labels"])
    indexes = []
    for i in range(data.shape[0]):
        if data.iloc[i, 0].endswith(".jpg"):
            indexes.append(i)
    test_dataset = dataset.PascalDataset(TEST_CSV,IMG_DIR,LABEL_DIR,20,indexes)
    test_loader = DataLoader(
        test_dataset,
        batch_size= BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        )
    net = model.AlexNet(20).to(device)
    net.load_state_dict(torch.load(CHECKPOINT, map_location=device)['model'])
    net.eval()
    mean_loss = []
    all_labels = []
    all_scores = []
    all_preds = []
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

            all_labels.append(label_batch)
            all_scores.append(output)
            all_preds.append(y_pred)

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    all_preds = np.concatenate(all_preds)
    test_metrics = utils.get_metrics(all_labels, all_preds)
    test_map, ap_list = utils.get_map(all_labels, all_scores)

    loss = sum(mean_loss) / len(mean_loss)
    print("Checkpoint:", CHECKPOINT)
    print("Test images:", len(all_labels))
    print("Loss: {:.4f}".format(loss))
    print("mAP: {:.4f}".format(test_map))
    print("Exact match acc: {:.4f}".format(test_metrics["acc"]))
    print("Hamming loss: {:.4f}".format(test_metrics["hamming_loss"]))
    print("Precision: {:.4f}".format(test_metrics["precision"]))
    print("Recall: {:.4f}".format(test_metrics["recall"]))
    print("F1: {:.4f}".format(test_metrics["f1"]))
    print("AP per class:")
    for i in range(len(ap_list)):
        print("  {:<14} {:.4f}".format(t.classes[i], ap_list[i]))

if __name__ == "__main__":
    test()