#!/bin/bash

# need to source this, run . ./install.sh
export PROVERO_HOME=$(realpath .)

# Models directory
# Include the pretrained models on ImageNet + pretrained models trained on
# smaller datasets like MNIST, CIFAR10
export PROVERO_MODELS_PATH=/tmp/

[ ! -d DATASET_DIR ] && mkdir -f DATASET_DIR

# Datasets (MNIST, CIFAR10, ImageNet) directory
export DATA_DIR=./DATASET_DIR
