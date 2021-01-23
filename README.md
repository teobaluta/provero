# PROVERO

PROVERO is a sampling-based tool to quantitatively estimate properties for neural networks with
probabilistic guarantees.  This work is by Teodora Baluta, Zheng Leong Chua, Kuldeep S. Meel and Prateek Saxena, as published in ICSE 2021. 


How to Install
--------------

We recommend you create a Python 3.6.8 (tested with 3.5.2 also) environment and install the requirements with `pip install -r
requirements.txt`. To set up paths run `. ./install.sh`. You will need to edit this file to your own system's paths.

How to Run
----------

To verify if there are less than `--thresh` adversarial samples within a L-p ball defined by a `--img_epsilon` of a given image with confidence `--delta`:

```
python nn_verify/ --netname <path_to_model> --dataset [mnist, cifar10, imagenet] --img_epsilon
<float> --thresh <threshold> --eta <error> --delta <confidence>
```
For example:

```
python nn_verify/ --netname /hdd/provero-private/eran_benchmark/nets/pytorch/mnist/convBigRELU__DiffAI.pyt --delta 0.01 --eta 0.01 --thresh 0.1
```

Benchmarks
----------

### 1. ERAN Benchmark

Neural nets pre-trained and provided by ERAN in `eran_benchmark/nets` (too big to upload on git -- use `eran_benchmark/download_nets.sh` script to dowload the neural nets) are in a specific format `.pyt` or `.tf`.

Test set for ERAN: 100 test images for MNIST and CIFAR10 (in `eran_benchmark/mnist_test.csv` and `eran_data/cifar10_test.csv`).

If `provero` is given a neural network in a `.pyt` or `.tf` format then it will run on the ERAN test set.
Supported datasets: MNIST and CIFAR10.

In the paper we ran the older ERAN version (commit `08fc8f63d60dce382624cf7e8307b1fe96420574`) that supports Tensorflow 1.11 models. Porting the model loading and parsing to Tensorflow 2.x is for future development work. 


### 2. Pretrained Benchmark

Pre-trained neural nets available in Keras trained on the ImageNet dataset - instead of a file select one of the option `VGG16,VGG19,ResNet18,ResNet50,AlexNet,DenseNet,Inception_v3` for `--netname`.

When you first run `provero` for the pre-trained models, these will be downloaded as `.h5` files and saved in `/tmp/<model_name>` folder by default.
Test set for pretrained: 100 images saved in `provero_benchmark/`.


### 3. PixelDP Model

Randomized ResNet20 neural net on the CIFAR10 dataset (downloaded from paper github).

How to Extend
-------------

1) What if I have my own models?

You can either convert models to `.pyt` or `.tf` format (as per ERAN tool) or you may add your own model loading or benchmarks. To add your own model loading code, follow the example of `nn_verify/eran_benchmark.py` to create a `nn_verify/<custommodel>_benchmark.py`.

2) Does it work on other properties?

Currently, we instantiate the property to the robustness property but extending it to other properties should not be too hard. For extending it to other properties, you should extend the `RandomVar` class in `nn_verify/provero.py` (see how we extend it for robustness in `RandomVarRobustness`). The only requirement is that the property class should have a `sample` method so that the algorithms in `nn_verify/provero.py`.


How to Cite
-----------
If you use PROVERO, please cite our work.

The benchmarks used in our evaluation can be found [here](TODO). More info on the project page, [PROVERO](https://teobaluta.github.io/PROVERO/).
