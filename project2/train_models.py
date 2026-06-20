import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from data_setup import load_data

def train_all_models():
    X_train, X_test, y_train, y_test, feature_names, species_labels = load_data()

    # Decision Trees
    tree_models = {}
    for max_leaves in range(2, 51):
        clf = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        tree_models[max_leaves] = {"model": clf, "acc": acc, "leaves": clf.get_n_leaves()}

    # Logistic Regression
    lr_models = {}
    for C in np.logspace(-3, 3, 50):
        lr = LogisticRegression(C=C, penalty='l1', solver='saga', max_iter=2000, random_state=42)
        lr.fit(X_train, y_train)
        acc = lr.score(X_test, y_test)
        nnz = int(np.count_nonzero(lr.coef_))
        key = round(float(C), 6)
        lr_models[key] = {"model": lr, "acc": acc, "complexity": nnz}

    return tree_models, lr_models, X_train, X_test, y_train, y_test, feature_names, species_labels


if __name__ == "__main__":
    tree_models, lr_models, _, _, _, _, _, _ = train_all_models()

    print("── Decision Trees ──")
    for k, v in list(tree_models.items())[:5]:
        print(f"  max_leaves={k:2d} | acc={v['acc']:.3f} | leaves={v['leaves']}")

    print("\n── Logistic Regression ──")
    for k, v in list(lr_models.items())[:5]:
        print(f"  C={k:.6f} | acc={v['acc']:.3f} | non-zero weights={v['complexity']}")
