import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import dataset
import model
import train as t

TEST_CSV = "data/PascalVOC/test.csv"
RUNS = [
    ["From scratch", t.OUTPUT_FILENAME],
    ["From scratch, no augmentation", t.OUTPUT_FILENAME + "_no-aug"],
    ["ImageNet pretrained", t.OUTPUT_FILENAME + "_pretrained"],
]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def plot_class_distribution():
    data = pd.read_csv(t.TRAIN_CSV, names=["images", "labels"])
    counts = [0] * t.NUM_CLASSES
    for i in range(data.shape[0]):
        classes = []
        with open(os.path.join(t.LABEL_DIR, data.iloc[i, 1])) as f:
            for l in f.readlines():
                c = int(l.split()[0])
                if c not in classes:
                    classes.append(c)
        for c in classes:
            counts[c] += 1

    names = [t.classes[i] for i in range(t.NUM_CLASSES)]
    order = np.argsort(counts)
    plt.figure(figsize=(8, 6))
    plt.barh([names[i] for i in order], [counts[i] for i in order], color="tab:blue")
    plt.xlabel("Number of training images containing the class")
    plt.title("Class distribution (train.csv, {} images)".format(data.shape[0]))
    plt.tight_layout()
    plt.savefig("plots/class_distribution.png")
    plt.close()
    print("saved plots/class_distribution.png")


def plot_ap_per_class():
    results = []
    for name, filename in RUNS:
        path = os.path.join(t.LOG_DIR, filename + "_test.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append([name, json.load(f)])
        else:
            print("skipping", name, "no file", path)

    names = [t.classes[i] for i in range(t.NUM_CLASSES)]
    order = np.argsort(results[0][1]["ap"])
    y = np.arange(t.NUM_CLASSES)
    height = 0.8 / len(results)

    plt.figure(figsize=(8, 9))
    for i in range(len(results)):
        name = results[i][0]
        ap = results[i][1]["ap"]
        label = "{} (mAP {:.3f})".format(name, results[i][1]["map"])
        plt.barh(y + i * height, [ap[j] for j in order], height=height, color=COLORS[i], label=label)
    plt.yticks(y + height * (len(results) - 1) / 2, [names[j] for j in order])
    plt.xlabel("Average precision on VOC2007 test")
    plt.xlim(0, 1)
    plt.title("AP per class")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/ap_per_class.png")
    plt.close()
    print("saved plots/ap_per_class.png")


def plot_run_comparison():
    plt.figure(figsize=(10, 4))
    for i in range(len(RUNS)):
        name = RUNS[i][0]
        path = os.path.join(t.LOG_DIR, RUNS[i][1] + ".json")
        if not os.path.exists(path):
            print("skipping", name, "no file", path)
            continue
        with open(path) as f:
            history = json.load(f)
        epochs = range(1, len(history["valid_map"]) + 1)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_map"], color=COLORS[i], linestyle="--")
        plt.plot(epochs, history["valid_map"], color=COLORS[i], label=name)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["train_loss"], color=COLORS[i], linestyle="--")
        plt.plot(epochs, history["valid_loss"], color=COLORS[i], label=name)

    plt.subplot(1, 2, 1)
    plt.title("mAP (dashed = train, solid = valid)")
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Loss (dashed = train, solid = valid)")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/run_comparison.png")
    plt.close()
    print("saved plots/run_comparison.png")


def plot_sample_predictions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = pd.read_csv(TEST_CSV, names=["images", "labels"])
    test_dataset = dataset.PascalDataset(TEST_CSV, t.IMG_DIR, t.LABEL_DIR, t.NUM_CLASSES, np.arange(0, data.shape[0]))

    net = model.AlexNet(t.NUM_CLASSES).to(device)
    net.load_state_dict(torch.load(os.path.join(t.CHECKPOINT_DIR, t.OUTPUT_FILENAME), map_location=device)["model"])
    net.eval()

    np.random.seed(3)
    indexes = np.random.choice(len(test_dataset), 12, replace=False)

    plt.figure(figsize=(12, 10))
    for i in range(len(indexes)):
        image, label = test_dataset[indexes[i]]
        with torch.no_grad():
            output = torch.sigmoid(net(image.unsqueeze(0).to(device)))[0].cpu().numpy()

        pred = [t.classes[j] for j in range(t.NUM_CLASSES) if output[j] >= 0.5]
        true = [t.classes[j] for j in range(t.NUM_CLASSES) if label[j] == 1]
        if sorted(pred) == sorted(true):
            color = "green"
        else:
            color = "red"

        show = np.transpose(image.numpy(), (1, 2, 0))
        show = show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        show = np.clip(show, 0, 1)

        plt.subplot(3, 4, i + 1)
        plt.imshow(show)
        plt.axis("off")
        plt.title("pred: {}\ntrue: {}".format(", ".join(pred) or "-", ", ".join(true)), fontsize=9, color=color)
    plt.tight_layout()
    plt.savefig("plots/sample_predictions.png")
    plt.close()
    print("saved plots/sample_predictions.png")


if __name__ == "__main__":
    plot_class_distribution()
    plot_ap_per_class()
    plot_run_comparison()
    plot_sample_predictions()
