import csv
import os
import numpy as np
import tensorflow as tf
import definitions
import keras
from keras.models import load_model
from pretrained_models import pretrained_tf

import eran.read_net_file
import clone_graph

PRETRAINED_NETS =['VGG16', 'VGG19', 'ResNet18', 'ResNet50', 'AlexNet',
                  'DenseNet', 'Inception_v3']

class LoadParams(object):
    '''
    Wrapper class for misc parameters of the loading/training of the model --
    from ERAN
    '''
    def __init__(self, is_trained_with_pytorch, is_conv, means, stds):
        self.is_trained_with_pytorch = is_trained_with_pytorch
        self.is_conv = is_conv
        self.means = means
        self.stds = stds

class NeuralNet_TF(object):
    def __init__(self, graph, input_names, output_names, batch_size):
        '''
        Wrapper class that loads a ERAN-saved model (.pyt,.tf) or an existing
        trained model.
        '''
        self.nn_batch_size = batch_size
        sess = tf.get_default_session()
        sess = tf.Session(graph=graph)
        self.graph = graph

        self.output_names = output_names
        if input_names is not None:
            self.input_names = input_names
        else:
            graph_def = self.graph.as_graph_def()
            x = tf.placeholder(np.float64, shape=[None, 784,], name='x')
            tf.import_graph_def(graph_def, {'x': x})
            self.input_names = 'x:0'

        self.x = self.graph.get_tensor_by_name(self.input_names)
        self.y = self.graph.get_tensor_by_name(self.output_names[0] + ':0')

        gpu_image = tf.placeholder(self.x.dtype, name="GPU_IMAGE")
        gpu_eps = tf.placeholder(self.x.dtype, name="GPU_EPS")

        self.sample_linf_tensor = tf.random_uniform((batch_size,)+tuple(self.x.shape[1:].as_list()), -1.0, 1.0, dtype=self.x.dtype) * gpu_eps + gpu_image

        # l2 d-ball uniform sampling
        x_shape_orig = tuple(self.x.shape.as_list())
        x_reshape = tf.reshape(self.x, (batch_size, -1))
        x_shape = (batch_size, np.prod(x_shape_orig[1:]))
        x_sample_shape = (batch_size, x_shape[1]+2)
        u = tf.random.normal(x_sample_shape, dtype=self.x.dtype)
        norm = tf.linalg.norm(u, axis=1)
        uniform_points = tf.reshape((u/norm[:,None])[:,:-2], (batch_size, )+x_shape_orig[1:])
        self.sample_l2_tensor = uniform_points * gpu_eps + gpu_image


    def sample(self, image, eps):
        return self.sample_tensor.eval(feed_dict={"GPU_EPS:0":eps, "GPU_IMAGE:0": image})

    def infer(self, image):
        sess = tf.get_default_session()
        image = np.asarray(image)
        output_tensor = self.graph.get_tensor_by_name(self.output_names[0] + ':0')
        out = sess.run(output_tensor, feed_dict = {self.input_names: image})
        return np.argmax(out, axis=1)


# use ERAN code to load models
def load_nn(netname, dataset, batch_size, batch_mode=False, distance_type='linf'):
    is_trained_with_pytorch = False
    is_saved_tf_model = False
    is_pretrained = False

    net_path = os.path.dirname(netname)
    filename, file_extension = os.path.splitext(netname)

    if file_extension == '.pyt':
        is_trained_with_pytorch = True
    elif file_extension == ".meta":
        is_saved_tf_model = True
    elif file_extension == '.hdf5':
        is_keras_model = True
    elif netname in PRETRAINED_NETS:
        is_pretrained = True
    elif file_extension != '.tf':
        print("file extension not supported")
        exit(1)

    input_names = None
    load_params = None

    print('[teo] is_saved_tf_model={}'.format(is_saved_tf_model))
    print('[teo] is_pretrained={}'.format(is_pretrained))

    if is_pretrained:
        net_path, input_tensor, output_tensor = pretrained_tf.keras_pretrained_model(netname)
        input_names = input_tensor.name
        output_names = [output_tensor.name[:-2]]
        print(input_names, output_names)
    if is_saved_tf_model or is_pretrained:
        print('Loading {}'.format(net_path))
        tf.logging.set_verbosity(tf.logging.ERROR)
        sess = keras.backend.get_session()
        meta_file_path = os.path.join(net_path, netname + '.meta')
        imported_meta = tf.train.import_meta_graph(meta_file_path)
        imported_meta.restore(sess, tf.train.latest_checkpoint(net_path))
        graph = sess.graph
    elif batch_mode:
        if dataset=='mnist':
            num_pixels = 784
        elif dataset=='cifar10':
            num_pixels = 3072
        else:
            num_pixels = 5
        x_batch, preds_batch, info, x, preds= clone_graph.read_eran_model(netname, batch_size,
                                                           num_pixels)
        is_conv, means, stds = info
        load_params = LoadParams(is_trained_with_pytorch, is_conv, means, stds)
    else:
        if dataset=='mnist':
            num_pixels = 784
        elif dataset=='cifar10':
            num_pixels = 3072
        else:
            num_pixels = 5
        model, is_conv, means, stds = eran.read_net_file.read_net(netname,
                                                                  [batch_size,
                                                                   num_pixels],
                                                                  is_trained_with_pytorch)
        print('model type()=' % model, type(model))
        print(model.__class__)
        if issubclass(model.__class__, tf.Tensor):
            output_names = [model.op.name]
        elif issubclass(model.__class__, tf.Operation):
            output_names = [model.name]
        elif issubclass(model.__class__, Sequential):
            output_names = [model.layers[-1].output.op.inputs[0].op.name]
            model        = model.layers[-1].output.op
        else:
            if issubclass(model.__class__, keras.engine.sequential.Sequential):
                output_names = [model.layers[-1].output.op.inputs[0].op.name]
                model        = model.layers[-1].output.op
            else:
                assert 0, "ERAN can't recognize this input"
        graph = model.graph
        load_params = LoadParams(is_trained_with_pytorch, is_conv, means, stds)

    if batch_mode:
        nn = BatchModeNN(x_batch, preds_batch, x, preds, batch_size)
    else:
        nn = NeuralNet_TF(graph, input_names, output_names, batch_size)

    if distance_type == 'linf':
        nn.sample_tensor = nn.sample_linf_tensor
    elif distance_type == 'l2':
        nn.sample_tensor = nn.sample_l2_tensor

    return nn, load_params

class BatchModeNN(object):
    def __init__(self, x_batch, preds_batch, x, preds, batch_size):
        self.x_batch = x_batch
        self.preds_batch = preds_batch
        self.preds = preds
        self.x = x
        self.batch_size = batch_size

        gpu_image = tf.placeholder(self.x.dtype, name="GPU_IMAGE")
        gpu_eps = tf.placeholder(self.x.dtype, name="GPU_EPS")

        self.sample_linf_tensor = tf.random_uniform((batch_size,)+tuple(self.x_batch.shape[1:].as_list()), -1.0, 1.0, dtype=self.x_batch.dtype) * gpu_eps + gpu_image

        # l2 d-ball uniform sampling
        x_batch_shape = tuple(self.x_batch.shape.as_list())
        x_reshape = tf.reshape(self.x_batch, (batch_size, -1))
        x_shape = x_reshape.shape.as_list()
        x_sample_shape = (x_shape[0], x_shape[1]+2)
        u = tf.random.normal(x_sample_shape, dtype=self.x_batch.dtype)
        norm = tf.linalg.norm(u,axis=1)
        random_points_flat = (u/norm[:,None])[:,:-2]
        uniform_points = tf.reshape(random_points_flat, (batch_size,)+x_batch_shape[1:])
        self.sample_l2_tensor = uniform_points * gpu_eps + gpu_image

    def sample(self, image, eps):
        return self.sample_tensor.eval(feed_dict={"GPU_EPS:0":eps, "GPU_IMAGE:0": image})

    def infer_batch(self, images):
        preds_out = self.preds_batch.eval(feed_dict={self.x_batch: images})
        return np.argmax(preds_out, axis=1)

    def infer(self, image):
        preds_out = self.preds.eval(feed_dict={self.x: image})
        return np.argmax(preds_out, axis=1)

