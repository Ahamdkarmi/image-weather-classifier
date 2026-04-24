import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

NPZ_PATH = "features_convnext_tiny_224.npz"
def analyze_errors(y_test, y_pred, idx_test, paths=None, knn=None, X_test=None, topk=10):
    wrong_mask = (y_pred != y_test)
    wrong_idx = idx_test[wrong_mask]
    print("\n-----------------------------")
    print("Performance analysis")
    print("-----------------------------")
    print(f"Total test samples: {len(y_test)}")
    print(f"Misclassified: {wrong_mask.sum()}")
    if wrong_mask.sum() == 0:
        print("No misclassifications found.")
        return
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
    print("\nExample misclassified samples:")
    show_n = min(topk, len(wrong_idx))
    for i in range(show_n):
        true_label = wrong_true[i]
        pred_label = wrong_pred[i]
        if paths is not None:
            print(f"{i+1:02d}) true={true_label}  pred={pred_label}  path={paths[wrong_idx[i]]}")
        else:
            print(f"{i+1:02d}) true={true_label}  pred={pred_label}")
    if knn is not None and X_test is not None:
        dists, _ = knn.kneighbors(X_test, n_neighbors=knn.n_neighbors, return_distance=True)
        avg_dist = dists.mean(axis=1)
        wrong_avg_dist = avg_dist[wrong_mask]
        order = np.argsort(wrong_avg_dist)
        print("\nTop confident WRONG predictions (closest neighbors):")
        for rank, j in enumerate(order[:show_n], 1):
            conf_like = wrong_avg_dist[j]
            true_label = wrong_true[j]
            pred_label = wrong_pred[j]
            if paths is not None:
                print(f"{rank:02d}) avg_dist={conf_like:.4f}  true={true_label}  pred={pred_label}  path={paths[wrong_idx[j]]}")
            else:
                print(f"{rank:02d}) avg_dist={conf_like:.4f}  true={true_label}  pred={pred_label}")
def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"]
    paths = data["paths"] if "paths" in data.files else None
    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y.shape)
    if paths is not None:
        print("Loaded paths shape:", paths.shape)
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.2,
        random_state=42,
        stratify=strat
    )
    for k in [1, 3]:
        print("\n=============================")
        print(f"KNN (k={k})")
        print("=============================")
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"KNN accuracy (k={k}): {acc:.4f}\n")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        analyze_errors(
            y_test=y_test,
            y_pred=y_pred,
            idx_test=idx_test,
            paths=paths,
            knn=knn,
            X_test=X_test,
            topk=10
        )
if __name__ == "__main__":
    main()
