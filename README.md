# Weather Classification from Images — CNN Features + Classical ML

## Overview

This project is part of **Assignment #3** for the *Machine Learning and Data Science (ENCS5341)* course at **Birzeit University**.

It focuses on classifying weather conditions from travel-destination images using a two-stage pipeline:

1. **Feature Extraction** — A frozen pre-trained ConvNeXt-Tiny CNN extracts high-level feature vectors from images and saves them to a compressed `.npz` file for efficient reuse.
2. **Classification** — Three classical ML models are trained and evaluated on the extracted features: KNN (baseline), Random Forest, and SVM.

---

## Objectives

* Extract rich image representations using a frozen pre-trained CNN (ConvNeXt-Tiny)
* Evaluate a KNN baseline with k=1 and k=3 on CNN features
* Train and tune Random Forest and SVM classifiers
* Perform error analysis to understand misclassification patterns
* Compare models using accuracy, precision, recall, and F1-score

---

## Project Structure

```
.
├── data/
│   ├── dataset.csv                      # Travel-destination dataset with image paths and labels
│   ├── features_convnext_tiny_224.npz   # Pre-extracted CNN feature vectors
│   └── Images_Link.txt                  # Image URL references
├── doc/
│   └── Report.pdf                       # Full project report
├── src/
│   ├── Frozen_CNN.py                    # Feature extraction using frozen ConvNeXt-Tiny
│   ├── KNN.py                           # KNN baseline classifier (k=1 and k=3)
│   ├── Random_Forest.py                 # Random Forest with grid search tuning
│   ├── SVM.py                           # SVM with RBF kernel and StandardScaler
│   └── main.py                          # EDA: dataset exploration and visualizations
└── README.md
```

---

## Feature Extraction

Features are extracted using `timm`'s **ConvNeXt-Tiny** model (pretrained on ImageNet) with the classification head removed (`num_classes=0`). All parameters are frozen — the model is used purely as a feature extractor.

* Input: Images resized to `224×224`, normalized with ImageNet mean/std
* Output: A feature matrix saved to `features_convnext_tiny_224.npz` (`X`, `y`, `paths`)
* Processing is done in batches of 32 for efficiency

---

## Models

### Baseline — KNN

* Evaluated with `k=1` and `k=3` using Euclidean distance
* Provides a simple distance-based reference point

### Proposed Model 1 — Random Forest

* 300 trees, entropy criterion, max depth 20
* Tuned via `GridSearchCV` over `n_estimators`, `max_depth`, and `min_samples_split`
* Includes detailed error analysis with confidence-based misclassification inspection

### Proposed Model 2 — SVM

* RBF kernel with `C=10.0`, `gamma="scale"`
* Features standardized using `StandardScaler` (via sklearn Pipeline)
* Probability estimates enabled for confidence analysis

---

## Results

| Model | Accuracy | F1 (weighted) | Note |
|---|---|---|---|
| KNN | 0.7136 | 0.70 | k=1 (best baseline) |
| Random Forest | 0.7042 | 0.61 | Tuned via grid search |
| SVM (RBF) | 0.6995 | 0.67 | Best single model |

**Best Model: KNN (k=1)**

---

## Dataset

The dataset was collected in Phase 1 of the assignment and contains travel-destination images with metadata including:

| Column | Description |
|---|---|
| `Image URL` | Path to the image file |
| `Weather` | Target label (Sunny, Cloudy, Snowy, Rainy, Clear) |
| `Season` | Season associated with the image |
| `Country` | Country of the destination |
| `Mood/Emotion` | Mood conveyed by the image |
| `Activity` | Activity shown |

A significant class imbalance was observed, with **Sunny** dominating (~700 samples) compared to minority classes like Rainy and Clear.

---

## Key Findings

* CNN features from a frozen ConvNeXt-Tiny enable competitive performance with classical ML
* All three models achieved comparable accuracy (~70%), suggesting a performance ceiling tied to dataset quality and class imbalance
* Misclassifications were concentrated in minority classes (Rainy, Clear), frequently predicted as Sunny
* High-confidence wrong predictions suggest the presence of ambiguous or mislabeled samples
* Hyperparameter tuning for Random Forest did not improve performance, indicating the bottleneck is data quality rather than model configuration

---

## How to Run

### 1. Install dependencies

```bash
pip install numpy pandas scikit-learn torch timm torchvision Pillow matplotlib
```

### 2. Extract CNN features (run once)

```bash
python src/Frozen_CNN.py
```

> Reads `data/dataset.csv`, processes images from the `Image URL` column, and saves `data/features_convnext_tiny_224.npz`.

### 3. Run EDA

```bash
python src/main.py
```

### 4. Train and evaluate models

```bash
python src/KNN.py
python src/Random_Forest.py
python src/SVM.py
```

> All three scripts load `features_convnext_tiny_224.npz` directly — no re-extraction needed.

---

## Visualizations Included

* Weather class distribution bar chart
* Season distribution bar chart
* Confusion matrices for all models
* Classification reports (Accuracy, Precision, Recall, F1)
* Top misclassified samples with confidence scores

---

## References

### Project Files

* [ Feature Extractor (Frozen_CNN.py)](src/Frozen_CNN.py)
* [ KNN Classifier (KNN.py)](src/KNN.py)
* [ Random Forest (Random_Forest.py)](src/Random_Forest.py)
* [ SVM Classifier (SVM.py)](src/SVM.py)
* [ EDA Script (main.py)](src/main.py)
* [ Dataset (dataset.csv)](data/dataset.csv)
* [ Image Links (Images_Link.txt)](data/Images_Link.txt)
* [ Report (Report.pdf)](doc/Report.pdf)

### Official Documentation

* [Scikit-learn](https://scikit-learn.org/stable/)
* [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
* [PyTorch](https://pytorch.org/docs/)
* [NumPy](https://numpy.org/doc/)
* [Pandas](https://pandas.pydata.org/docs/)
* [Matplotlib](https://matplotlib.org/stable/contents.html)

---

## Authors

* **Author:** Ahmad Karmi
* **Course:** Machine Learning and Data Science — ENCS5341
* **Institution:** Birzeit University

---

## Notes

This project was developed as part of a university machine learning assignment. The dataset was self-collected and is used for educational purposes only.
