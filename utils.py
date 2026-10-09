import sklearn.metrics as metrics 
import pandas as pd
import train as t
import torch
import matplotlib.pyplot as plt
import numpy as np
import dataset 
def get_metrics(label_batch, y_pred):

    exact_match = metrics.accuracy_score(
        label_batch, y_pred, normalize=True
        )  # how many predictions match the labels exactly
    hamming_loss = metrics.hamming_loss(
            label_batch, y_pred
        )  # error rate, 0 is good 1 is bad
    precision = metrics.precision_score(
            label_batch, y_pred, average="samples", zero_division=1
        )  # how well it avoids predicting negatives as positive. 1 is best
    recall = metrics.recall_score(
            label_batch, y_pred, average="samples"
        )  # how many positives it predicted as positive. 1 is best
    f_1 = metrics.f1_score(
            label_batch, y_pred, average="samples"
        )  # balance of precision and recall, 1 is good 0 is bad

    return {
        "acc": exact_match,
        "hamming_loss": hamming_loss,
        "precision": precision,
        "recall": recall,
        "f1": f_1,
    }

def get_map(labels, scores):
    ap_list = []
    total = 0
    count = 0
    for c in range(labels.shape[1]):
        if labels[:, c].sum() > 0:
            ap = metrics.average_precision_score(labels[:, c], scores[:, c])
            total += ap
            count += 1
        else:
            ap = 0
        ap_list.append(ap)
    return total / count, ap_list

def plot_class_dist(set):
    
    classes = []
    dist = {}

    for i, label in set:
        one_label = torch.nonzero(label)
        for c in one_label:
            classes.append(c.item())    
    for i in range(20):
        dist[i] = classes.count(i)
        
    x = list(dist.keys())
    y = list(dist.values())
    plt.bar(x,y,width=0.2)
    plt.xticks(np.arange(0,21,1))
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv(t.TRAIN_CSV, names=["images", "labels"])
    fold_idx = np.arange(0,int((data.shape[0] * 2) / 3))
    fold_idx_2 = np.arange(int((data.shape[0] * 2) / 3), data.shape[0])
    ds = dataset.PascalDataset(t.TRAIN_CSV,t.IMG_DIR,t.LABEL_DIR, t.NUM_CLASSES,fold_idx)
    ds_2 = dataset.PascalDataset(t.TRAIN_CSV, t.IMG_DIR, t.LABEL_DIR, t.NUM_CLASSES, fold_idx_2)
    plot_class_dist(ds)
    plot_class_dist(ds_2)