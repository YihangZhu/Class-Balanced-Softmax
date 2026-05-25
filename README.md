# Class-Balanced Softmax (CBS)
This repository contains the implementation and data for Class-Balanced Softmax (CBS).

## 📊 Data Preparation

1. **Download the datasets** using the following links:
   * [ImageNet (Object Localization Challenge)](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
   * [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018)
   * [Places365](https://www.kaggle.com/datasets/benjaminkz/places365)
   * [LVIS](https://www.lvisdataset.org/dataset)

2. **Update the local dataset paths** in your code:
   * Modify the paths in `datasets/iip_dataloaders.py` on lines **27, 46, 63, and 73**.
   * Modify the path in `datasets/coco_lvis.py` on line **300**.

---

## ⚙️ Environment & Setup

Before running the scripts, prepare your Python environment. You can refer to this [Python Venv Training Tutorial](https://sites.google.com/view/zhuyihang/ml-training-tutorial/train-on-the-hpcs/python-venv?authuser=0) for detailed guidance on setting up virtual environments on HPCS.

Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Training & Evaluation

### Training Models

To train a model from scratch using Distributed Data Parallel (DDP), execute the following command with your desired configuration file:

```bash
python ddp_main.py --cfg ['config/lvis/lvis_v1_cbs.yaml']
```

### Evaluating Checkpoints

All the results, including the checkpoints, can be downloaded from [this link](https://www.icloud.com/iclouddrive/0829G1RulTZsYWJoBRaGSf0eg#cbs%5Fpaper%5Fresults).

To evaluate a specific checkpoint (for example, on `LVIS`), run:

```bash
python ddp_main.py --cfg config/lvis/lvis_v1_cbs.yaml --test_checkpoint 1 --checkpoint /path/to/checkpoint
```

---

## 📈 Generating Figures

You can replicate and generate all the figures featured in the paper by running the automated evaluation script:

```bash
python generate_figures.py
```

---
