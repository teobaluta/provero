# PROVERO

PROVERO is a sampling-based tool to quantitatively estimate properties for neural networks with
probabilistic guarantees.  This work is by Teodora Baluta, Zheng Leong Chua, Kuldeep S. Meel and Prateek Saxena, as published in ICSE 2021. 


How to Install
--------------

We recommend you create a Python 3.6.8 (tested with 3.5.2 also) environment and install the requirements with `pip install -r
requirements.txt`. To set up paths run `. ./install.sh`. You will need to edit this file to your own system's paths.

##### Environment Variables

The `install.sh` sets the following list of paths/environment vairables

- `PROVERO_HOME`: path to the project's root directory
- `DATA_DIR`: path to directory containing `ImageNet` dataset. For the pretrained models, it will try to load either from the `imagenet_test.pkl`, `imagenet_test_inception.pkl` or by default create the folder `DATASET_DIR` to load the dataset from it. If you have an existing download for the `ImageNet` dataset, change this variable to point to it  
- `PROVERO_MODELS_PATH`: where to save to/load from pretrained models (as available from `keras.applications` package). By default, this is set to `/tmp/`

##### Other paths (in definitions.py)
Most of these variables are set with respect to the environment variables.
- `NB_TEST_SAMPLES`: number of test samples to run `provero` on
- `LOGS_PATH`: creates and saves logs from running `provero` at this path, by default set to `$PROVERO_HOME/logs`
- `ERAN_TEST_PATH`: the path to the test set for the ERAN models, by default set to `$PROVERO_HOME/eran_benchmark`
- `PROVERO_TEST_PATH`: the path to the test set for the pretrained models
- `PIXELDP_NETS`: the name of the PixelDP model
- `PRETRAINED_NETS`: list of names of supported pretrained nets 

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

Excerpt from the expected output (omitted are Tensorflow messages)
```
...
W shape (3, 3, 1, 32)
Conv2D {'filters': 32, 'kernel_size': [3, 3], 'input_shape': [28, 28, 1], 'stride': [1, 1], 'padding': 1} W.shape: (3, 3, 1, 32) b.shape: (32,)
        OutShape:  (1, 28, 28, 32)
W shape (4, 4, 32, 32)
Conv2D {'filters': 32, 'kernel_size': [4, 4], 'input_shape': [28, 28, 32], 'stride': [2, 2], 'padding': 1} W.shape: (4, 4, 32, 32) b.shape: (32,)
        OutShape:  (1, 14, 14, 32)
W shape (3, 3, 32, 64)
Conv2D {'filters': 64, 'kernel_size': [3, 3], 'input_shape': [14, 14, 32], 'stride': [1, 1], 'padding': 1} W.shape: (3, 3, 32, 64) b.shape: (64,)
        OutShape:  (1, 14, 14, 64)
W shape (4, 4, 64, 64)
Conv2D {'filters': 64, 'kernel_size': [4, 4], 'input_shape': [14, 14, 64], 'stride': [2, 2], 'padding': 1} W.shape: (4, 4, 64, 64) b.shape: (64,)
        OutShape:  (1, 7, 7, 64)
ReLU
        OutShape:  (1, 512)
        WShape:  (3136, 512)
        BShape:  (512,)
ReLU
        OutShape:  (1, 512)
        WShape:  (512, 512)
        BShape:  (512,)
Affine
        OutShape:  (1, 10)
        WShape:  (512, 10)
        BShape:  (10,)
mean=[0.1307],stds=[0.3081]
...
RandomVar initialized - correct_label=7; eps=0.1
Running adaptive_assert!
Run for 144 samples
s:0
Sampling took 0.465954065322876 sec
img 0 #adversarial < 0.1: Yes 0.4702014923095703 sec
RandomVar initialized - correct_label=2; eps=0.1
Running adaptive_assert!
Run for 144 samples
s:0
Sampling took 0.4389660358428955 sec
img 1 #adversarial < 0.1: Yes 0.44341611862182617 sec
RandomVar initialized - correct_label=1; eps=0.1
Running adaptive_assert!
Run for 144 samples

....

```


Benchmarks
----------

### 1. ERAN Benchmark (BM1)

#### Models
Neural nets pre-trained and provided by ERAN in `eran_benchmark/nets` (too big to upload on git -- use `eran_benchmark/download_nets.sh` script to dowload the neural nets) are in a specific format `.pyt` or `.tf`.

If `provero` is given a neural network in a `.pyt` or `.tf` format then it will run on the ERAN test set.
Supported datasets: MNIST and CIFAR10.

In the paper we ran the older ERAN version (commit `08fc8f63d60dce382624cf7e8307b1fe96420574`) that supports Tensorflow 1.11 models. Porting the model loading and parsing to Tensorflow 2.x is for future development work. 

#### Test set
Test set for ERAN: 100 test images for MNIST and CIFAR10 (in `eran_benchmark/mnist_test.csv` and `eran_data/cifar10_test.csv`). You may run it on other inputs by adding your input in the ERAN test set format. More information is available on the [ERAN](https://github.com/eth-sri/eran) Github repo.


### 2.  Larger Models (BM2)

#### Models

Pre-trained neural nets available in Keras trained on the ImageNet dataset. To run `provero` on these, instead of a file select one of the option `VGG16,VGG19,ResNet50,DenseNet,Inception_v3` for `--netname`.

When you first run `provero` for the pre-trained models, these will be downloaded as `.h5` files and saved in `/tmp/<model_name>` folder by default.

#### Test Set
Test set for pretrained: 100 images saved in `provero_benchmark/`. We recommend downloading the `.pkl` files containing the test set used in the evaluation first as `ImageNet` is a large dataset that requires around 150GB.


### 3. PixelDP Model

Randomized ResNet20 neural net on the CIFAR10 dataset (downloaded from [pixeldp](https://github.com/columbia/pixeldp) repo).

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
