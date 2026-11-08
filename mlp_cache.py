"""Train (or load from models/) the MLP used in cross_dataset.py, with the same split and seed,
so later experiments start from exactly the same network."""
import os, pickle, warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
import datasets

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def splits(name, seed, n_val=5000):
    Xtr, ytr, Xte, yte = datasets.load(name)
    perm = np.random.default_rng(seed).permutation(len(Xtr))
    va, tr = perm[:n_val], perm[n_val:]
    f = lambda X: X.astype(np.float64) / 255.0
    return f(Xtr[tr]), ytr[tr], f(Xtr[va]), ytr[va], f(Xte), yte


def trained_mlp(name, seed, hidden=256, epochs=30, folder="models"):
    path = os.path.join(folder, f"mlp_{name}_seed{seed}.pkl")
    Xtrain, ytrain, Xval, yval, Xtest, ytest = splits(name, seed)
    if os.path.exists(path):
        clf = pickle.load(open(path, "rb"))
    else:
        clf = MLPClassifier(hidden_layer_sizes=(hidden,), batch_size=256, max_iter=epochs,
                            alpha=1e-4, random_state=seed).fit(Xtrain, ytrain)
        os.makedirs(folder, exist_ok=True)
        pickle.dump(clf, open(path, "wb"))
    return clf, (Xtrain, ytrain, Xval, yval, Xtest, ytest)
