import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

NPZ_PATH = "features_convnext_tiny_224.npz"
def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"]
    paths = data["paths"] if "paths" in data.files else None
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    if paths is not None:
        print("paths shape:", paths.shape)
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.2, random_state=42, stratify=strat
    )
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM (RBF) accuracy: {acc:.4f}\n")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    wrong_mask = (y_pred != y_test)
    wrong_indices = idx_test[wrong_mask]
    print("\n-----------------------------")
    print("Performance analysis")
    print("-----------------------------")
    print(f"Total test samples: {len(y_test)}")
    print(f"Misclassified: {wrong_mask.sum()}")
    wrong_true = y_test[wrong_mask]
    wrong_pred = y_pred[wrong_mask]
    print("\nMost frequent TRUE labels among mistakes:")
    for lab, cnt in Counter(wrong_true).most_common(10):
        print(f"  {lab}: {cnt}")
    print("\nMost frequent PREDICTED labels among mistakes:")
    for lab, cnt in Counter(wrong_pred).most_common(10):
        print(f"  {lab}: {cnt}")
    pair_counts = Counter(zip(wrong_true, wrong_pred))
    print("\nTop confusions (true -> predicted):")
    for (t, p), cnt in pair_counts.most_common(10):
        print(f"  {t} -> {p}: {cnt}")
    proba = model.predict_proba(X_test)
    pred_conf = proba.max(axis=1)
    wrong_conf = pred_conf[wrong_mask]
    if wrong_conf.size > 0:
        topk = np.argsort(-wrong_conf)[:10]
        print("\nTop confident WRONG predictions (potential label/data issues):")
        for rank, j in enumerate(topk, 1):
            true_label = wrong_true[j]
            pred_label = wrong_pred[j]
            conf = wrong_conf[j]
            if paths is not None:
                img_path = paths[wrong_indices[j]]
                print(f"{rank:02d}) conf={conf:.3f}  true={true_label}  pred={pred_label}  path={img_path}")
            else:
                print(f"{rank:02d}) conf={conf:.3f}  true={true_label}  pred={pred_label}")
    train_counts = Counter(y_train)
    rare = [lab for lab, cnt in train_counts.items() if cnt < 10]
    if rare:
        print("\nRare classes in TRAIN (<10 samples) that may cause errors:")
        print("  " + ", ".join(sorted(rare)))

if __name__ == "__main__":
    main()
