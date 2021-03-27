#!/bin/bash

# need to source this, run . ./install.sh
export PROVERO_HOME=$(realpath .)

# Models directory
# Include the pretrained models on ImageNet + pretrained models trained on
# smaller datasets like MNIST, CIFAR10
export PROVERO_MODELS_PATH=/tmp/

[ ! -d DATASET_DIR ] && mkdir -p DATASET_DIR

# Datasets (MNIST, CIFAR10, ImageNet) directory
export DATA_DIR=./DATASET_DIR
[ ! -d pixeldp ] && git clone https://github.com/columbia/pixeldp
cd pixeldp
# TODO check if its already downloaded
[ ! -d cifar10 ] && wget http://www.cs.columbia.edu/~mathias/pixeldp/cifar10.zip && \
	unzip cifar10.zip && \
	mkdir -p trained_models/pixeldp_resnet_attack_norm_l2_size_0.1_1_prenoise_layers_sensitivity_l2_scheme_bound_activation_postnoise && \
	cp cifar10/* trained_models/pixeldp_resnet_attack_norm_l2_size_0.1_1_prenoise_layers_sensitivity_l2_scheme_bound_activation_postnoise
cd ..

# TODO move the pixeldp inside the project
export PIXELDP_HOME=$(pwd)/pixeldp
