#!/bin/bash

# abort entire script on error
set -e

if [ ! -d 'snapshot' ]; then
      mkdir snapshot
fi;

if [ ! -d 'data' ]; then
      mkdir data
fi;

# train base model on svhn
python tools/train.py svhn train lenet "lenet_svhn$1" \
       --iterations 20000 \
       --gpu $1 \
       --batch_size 128 \
       --display 10 \
       --lr 0.001 \
       --snapshot 1000 \
       --solver adam
       --stepsize 5000

# run adda svhn->mnist
python tools/train_adda.py svhn:train mnist:train lenet "adda_lenet_svhn_mnist$1" \
       --iterations 20000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.0002 \
       --snapshot 1000 \
       --gpu $1 \
       --weights "snapshot/lenet_svhn1" \
       --adversary_relu \
       --solver adam
       --stepsize 5000
# evaluate trained models
echo 'Source only baseline:'
python tools/eval_classification.py mnist train lenet "snapshot/lenet_svhn$1"

echo 'ADDA':
python tools/eval_classification.py mnist train lenet "snapshot/adda_lenet_svhn_mnist$1"
