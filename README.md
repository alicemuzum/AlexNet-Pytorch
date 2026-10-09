# AlexNet from Scratch on Pascal VOC

A PyTorch implementation of AlexNet ([Krizhevsky et al., 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)), trained **from scratch** (no ImageNet pretraining) for **multi-label image classification** on Pascal VOC. Each image can contain several of the 20 object classes, so the network predicts an independent probability per class.

**Test set result: 54.2% mAP** on the 4,952 images of VOC2007 test.

## Results

Trained for 60 epochs on a single RTX 3060 Laptop GPU (about 30 seconds per epoch, ~30 minutes total).

| Metric (VOC2007 test, 4,952 images) | Score |
| --- | --- |
| mAP | **0.542** |
| F1 (sample-averaged) | 0.534 |
| Precision (sample-averaged) | 0.776 |
| Recall (sample-averaged) | 0.523 |
| Hamming loss | 0.050 |
| Exact match accuracy | 0.370 |

Predictions use a 0.5 threshold on the sigmoid outputs. mAP is the mean of per-class average precision computed with scikit-learn (non-interpolated, so not exactly the official VOC2007 11-point metric).

### Training curves

![Loss and mAP curves](plots/wdecay-5e-05_epoch-60.png)

- Train and validation follow each other closely until about epoch 30, after that the model starts to overfit and the gap grows.
- The learning rate is divided by 10 at epoch 45, which gives the visible jump in both curves. After that validation mAP stays flat around 0.55.
- Best validation mAP was 0.549 (epoch 56), final was 0.546.

### Average precision per class

| Class | AP | Class | AP |
| --- | --- | --- | --- |
| person | 0.866 | bird | 0.504 |
| car | 0.769 | dog | 0.489 |
| aeroplane | 0.736 | tv_monitor | 0.488 |
| train | 0.724 | diningtable | 0.457 |
| horse | 0.712 | chair | 0.446 |
| motorbike | 0.704 | sofa | 0.434 |
| bicycle | 0.628 | sheep | 0.404 |
| bus | 0.566 | cow | 0.329 |
| boat | 0.558 | potted_plant | 0.317 |
| cat | 0.529 | bottle | 0.188 |

Classes that are common and usually large in the image (person, car, aeroplane) are the easiest. Small objects (bottle, potted plant) and classes that look alike (cow and sheep) are the hardest.

## Model

Input images are 227×227. The paper says 224, but with an 11×11 kernel and stride 4 only 227 gives the 55×55 feature map described in the paper.

| Layer | Configuration | Output |
| --- | --- | --- |
| Input | | 3 × 227 × 227 |
| Conv1 | 96 filters 11×11, stride 4, ReLU, LRN, max pool 3/2 | 96 × 27 × 27 |
| Conv2 | 256 filters 5×5, pad 2, ReLU, LRN, max pool 3/2 | 256 × 13 × 13 |
| Conv3 | 384 filters 3×3, pad 1, ReLU | 384 × 13 × 13 |
| Conv4 | 384 filters 3×3, pad 1, ReLU | 384 × 13 × 13 |
| Conv5 | 256 filters 3×3, pad 1, ReLU, max pool 3/2 | 256 × 6 × 6 |
| FC6 | Dropout 0.5, 9216 → 4096, ReLU | 4096 |
| FC7 | Dropout 0.5, 4096 → 4096, ReLU | 4096 |
| FC8 | 4096 → 20 | 20 |

58.4M parameters. The 20 outputs go through a sigmoid and are trained with binary cross entropy (one binary problem per class).

Differences from the paper:
- Trained on ~11k images instead of 1.2M ImageNet images, and for multi-label instead of single-label classification.
- Single GPU, so the convolutions are not split into two groups.
- Adam optimizer instead of SGD with momentum.
- Default PyTorch weight initialization instead of the paper's N(0, 0.01) weights and bias 1.
- Augmentation is random 227 crops from 256 and horizontal flips. No PCA color augmentation and no 10-crop testing.

## Dataset

Pascal VOC in YOLO label format, from Kaggle: [aladdinpersson/pascalvoc-yolo](https://www.kaggle.com/datasets/aladdinpersson/pascalvoc-yolo).

- **Train:** VOC2007 trainval + VOC2012 trainval, 16,551 images. Split randomly (seed 0) into 11,034 for training and 5,517 for validation.
- **Test:** VOC2007 test, 4,952 images.

Every label file lists the objects in the image (`class x y w h`). Only the class index is used, turned into a 20-dim multi-hot vector.

Download and extract it into `data/PascalVOC`:

```bash
mkdir -p data/PascalVOC
curl -L -o data/pascalvoc-yolo.zip https://www.kaggle.com/api/v1/datasets/download/aladdinpersson/pascalvoc-yolo
unzip -q data/pascalvoc-yolo.zip -d data/PascalVOC
rm data/pascalvoc-yolo.zip
```

```
data/PascalVOC/
├── images/       000005.jpg, ...
├── labels/       000005.txt, ...
├── train.csv     image,label pairs
└── test.csv
```

## Usage

Requires Python 3.10. With [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

(or `pip install -r requirements.txt` in any Python 3.10 environment)

Train:

```bash
python train.py
```

Runs are reproducible: the model init, data shuffling, augmentation and train/validation split are all seeded. Hyperparameters are constants at the top of `train.py` (batch size 64, 60 epochs, Adam lr 1e-4, weight decay 5e-5, lr × 0.1 at epoch 45). A run saves:
- `models/wdecay-5e-05_epoch-60`: checkpoint after the last epoch
- `models/wdecay-5e-05_epoch-60_best`: checkpoint from the epoch with the best validation mAP
- `log/wdecay-5e-05_epoch-60.json`: metrics for every epoch
- `plots/wdecay-5e-05_epoch-60.png`: loss and mAP curves

Evaluate on the test set (uses the checkpoint from `train.py` by default):

```bash
python test.py
python test.py models/wdecay-5e-05_epoch-60_best
```

Other scripts:
- `ut.py`: checks the model output shape and the dataset shapes, shows some training images with their labels
- `utils.py`: plots the class distribution of the train/validation split
- `overfit.py`: sanity check, trains a BatchNorm version of AlexNet on 16 images to see that it can memorize them

## Project structure

```
├── model.py      AlexNet
├── dataset.py    PascalDataset, image transforms and multi-hot labels
├── train.py      training loop, validation, hyperparameters
├── test.py       evaluation on VOC2007 test
├── utils.py      metrics (mAP, F1, ...) and class distribution plot
├── ut.py         shape checks and visualization
├── overfit.py    overfitting sanity check
└── plots/        figures
```

## References

- A. Krizhevsky, I. Sutskever, G. E. Hinton. *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012.
- M. Everingham et al. *The Pascal Visual Object Classes (VOC) Challenge.* IJCV 2010.

## License

MIT
