#!/usr/bin/env bash
# Download MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 into data/ (about 400 MB).
set -euo pipefail
mkdir -p data/mnist data/fashion
cd data

MNIST=https://ossci-datasets.s3.amazonaws.com/mnist
FASHION=http://fashion-mnist.s3-website.eu-central-1.amazonaws.com
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  [ -f mnist/$f.gz ]   || curl -L -o mnist/$f.gz   $MNIST/$f.gz
  [ -f fashion/$f.gz ] || curl -L -o fashion/$f.gz $FASHION/$f.gz
done

[ -d cifar-10-batches-py ] || { curl -L -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz; tar xzf cifar-10-python.tar.gz; rm cifar-10-python.tar.gz; }
[ -d cifar-100-python ]    || { curl -L -O https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz; tar xzf cifar-100-python.tar.gz; rm cifar-100-python.tar.gz; }
echo "done: $(du -sh . | cut -f1) in data/"
