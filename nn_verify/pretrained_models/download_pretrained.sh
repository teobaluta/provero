#!/bin/bash

# following the instructions here
# https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

CHECKPOINT_DIR=/tmp/checkpoints
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt ${CHECKPOINT_DIR}
rm vgg_16_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
tar -xvf vgg_19_2016_08_28.tar.gz
mv vgg_19.ckpt ${CHECKPOINT_DIR}
rm vgg_19_2016_08_28.tar.gz
