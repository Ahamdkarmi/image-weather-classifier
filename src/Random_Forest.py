import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

NPZ_PATH = "features_convnext_tiny_224.npz"
def performance_analysis(y_test, y_pred, idx_test, paths=None, proba=None, topk=10):
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
        if paths is not None:
            print(f"{i+1:02d}) true={wrong_true[i]}  pred={wrong_pred[i]}  path={paths[wrong_idx[i]]}")
        else:
            print(f"{i+1:02d}) true={wrong_true[i]}  pred={wrong_pred[i]}")
    if proba is not None:
        pred_conf = proba.max(axis=1)
        wrong_conf = pred_conf[wrong_mask]
        order = np.argsort(-wrong_conf)[:show_n]
        print("\nTop confident WRONG predictions (potential label/data issues):")
        for rank, j in enumerate(order, 1):
            conf = wrong_conf[j]
            if paths is not None:
                print(f"{rank:02d}) conf={conf:.3f}  true={wrong_true[j]}  pred={wrong_pred[j]}  path={paths[wrong_idx[j]]}")
            else:
                print(f"{rank:02d}) conf={conf:.3f}  true={wrong_true[j]}  pred={wrong_pred[j]}")
def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"]
    paths = data["paths"] if "paths" in data.files else None
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.2,
        random_state=29,
        stratify=strat
    )
    model = RandomForestClassifier(
        n_estimators=300,
        criterion="entropy",
        max_depth=20,
        min_samples_split=5,
        random_state=23,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    proba = model.predict_proba(X_test)
    performance_analysis(y_test, y_pred, idx_test, paths=paths, proba=proba, topk=10)
    hyperparameter = {
        "n_estimators": [300, 500],
        "max_depth": [20, 50],
        "min_samples_split": [2, 5],
        "criterion": ["entropy"]
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=23, n_jobs=-1),
        hyperparameter,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("\nBest parameters:", grid.best_params_)
    best_model = grid.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    print("Tuned RF Accuracy:", accuracy_score(y_test, y_pred_tuned))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    proba_tuned = best_model.predict_proba(X_test)
    performance_analysis(y_test, y_pred_tuned, idx_test, paths=paths, proba=proba_tuned, topk=10)

if __name__ == "__main__":
    main()
