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

DIR=$(pwd)
cd $HOME
git clone https://github.com/columbia/pixeldp
cd pixeldp
wget  http://www.cs.columbia.edu/~mathias/pixeldp/cifar10.zip
unzip cifar10.zip
mkdir -p trained_models/pixeldp_resnet_attack_norm_l2_size_0.1_1_prenoise_layers_sensitivity_l2_scheme_bound_activation_postnoise
cp cifar10/* trained_models/pixeldp_resnet_attack_norm_l2_size_0.1_1_prenoise_layers_sensitivity_l2_scheme_bound_activation_postnoise
cd $DIR

export PIXELDP_HOME=$HOME/pixeldp
