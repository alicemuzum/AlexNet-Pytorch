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
        )  # Ne kadar prediction tamı tamına aynısı
    hamming_loss = metrics.hamming_loss(
            label_batch, y_pred
        )  # Error rate, 0 iyi 1 kötü
    precision = metrics.precision_score(
            label_batch, y_pred, average="samples", zero_division=1
        )  # Ne kadar negative olanı positive tahmin etmiyor. 1 en iyi
    recall = metrics.recall_score(
            label_batch, y_pred, average="samples"
        )  # Ne kadar positive olanlara positive demiş. 1 en iyi
    f_1 = metrics.f1_score(
            label_batch, y_pred, average="samples"
        )  # precision ve recallın oranı 1 iyi 0 kötü

    return {
        "acc": exact_match,
        "hamming_loss": hamming_loss,
        "precision": precision,
        "recall": recall,
        "f1": f_1,
    }

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