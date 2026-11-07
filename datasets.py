"""
Loaders for MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100.

Each loader returns (Xtr, ytr, Xte, yte): images as uint8 arrays of shape (N, D), flattened
(CIFAR in height-width-channel order), and integer labels. The official train/test split is used.

Put the files in data/ (get_data.sh downloads them):
    data/mnist/      train-images-idx3-ubyte(.gz) ... t10k-labels-idx1-ubyte(.gz)
    data/fashion/    the same four files for Fashion-MNIST
    data/cifar-10-batches-py/   official python version of CIFAR-10
    data/cifar-100-python/      official python version of CIFAR-100
CIFAR also loads from image folders, data/cifar10/{train,test}/<class>/*.png (and data/cifar100/...).
The first load is cached as data/cache/<name>.npz.
"""
import gzip, os, pickle
import numpy as np

NAMES = ["mnist", "fashion", "cifar10", "cifar100"]
PRETTY = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
N_CLASSES = {"mnist": 10, "fashion": 10, "cifar10": 10, "cifar100": 100}


def _open(path):
    if os.path.exists(path):
        return open(path, "rb")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rb")
    raise FileNotFoundError(path + "(.gz) not found; run get_data.sh")


def _idx(path):
    with _open(path) as f:
        data = f.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _mnist_like(folder):
    Xtr = _idx(os.path.join(folder, "train-images-idx3-ubyte")).reshape(-1, 784)
    ytr = _idx(os.path.join(folder, "train-labels-idx1-ubyte")).astype(np.int64)
    Xte = _idx(os.path.join(folder, "t10k-images-idx3-ubyte")).reshape(-1, 784)
    yte = _idx(os.path.join(folder, "t10k-labels-idx1-ubyte")).astype(np.int64)
    return Xtr, ytr, Xte, yte


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _chw_to_hwc(X):
    return X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(-1, 3072)


def _image_folders(folder):
    from PIL import Image
    out = []
    for split in ("train", "test"):
        classes = sorted(os.listdir(os.path.join(folder, split)))
        X, y = [], []
        for c, name in enumerate(classes):
            d = os.path.join(folder, split, name)
            for fn in sorted(os.listdir(d)):
                X.append(np.asarray(Image.open(os.path.join(d, fn)).convert("RGB"), np.uint8).reshape(-1))
                y.append(c)
        out += [np.stack(X), np.array(y, np.int64)]
    return out


def _cifar10(root):
    py = os.path.join(root, "cifar-10-batches-py")
    if os.path.isdir(py):
        tr = [_unpickle(os.path.join(py, f"data_batch_{i}")) for i in range(1, 6)]
        te = _unpickle(os.path.join(py, "test_batch"))
        Xtr = _chw_to_hwc(np.concatenate([b["data"] for b in tr]))
        ytr = np.concatenate([b["labels"] for b in tr]).astype(np.int64)
        return Xtr, ytr, _chw_to_hwc(te["data"]), np.array(te["labels"], np.int64)
    return _image_folders(os.path.join(root, "cifar10"))


def _cifar100(root):
    py = os.path.join(root, "cifar-100-python")
    if os.path.isdir(py):
        tr, te = _unpickle(os.path.join(py, "train")), _unpickle(os.path.join(py, "test"))
        return (_chw_to_hwc(tr["data"]), np.array(tr["fine_labels"], np.int64),
                _chw_to_hwc(te["data"]), np.array(te["fine_labels"], np.int64))
    return _image_folders(os.path.join(root, "cifar100"))


def load(name, root="data"):
    cache = os.path.join(root, "cache", name + ".npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    if name == "mnist":
        Xtr, ytr, Xte, yte = _mnist_like(os.path.join(root, "mnist"))
    elif name == "fashion":
        Xtr, ytr, Xte, yte = _mnist_like(os.path.join(root, "fashion"))
    elif name == "cifar10":
        Xtr, ytr, Xte, yte = _cifar10(root)
    elif name == "cifar100":
        Xtr, ytr, Xte, yte = _cifar100(root)
    else:
        raise ValueError(name)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    for n in NAMES:
        Xtr, ytr, Xte, yte = load(n)
        print(f"{PRETTY[n]:14s} train {Xtr.shape} test {Xte.shape} classes {len(np.unique(ytr))}")
