from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

import eran.read_net_file
import eran_benchmark

import os
import sys
import time
import pickle
import ntpath

def clone_subgraph(outputs, mappings, clone_scope=''):
    NON_REPLICABLE = {'Variable', 'VariableV2', 'AutoReloadVariable',
                      'MutableHashTable', 'MutableHashTableV2',
                      'MutableHashTableOfTensors', 'MutableHashTableOfTensorsV2',
                      'MutableDenseHashTable', 'MutableDenseHashTableV2',
                      'VarHandleOp', 'BoostedTreesEnsembleResourceHandleOp'}
    ops = ge.get_backward_walk_ops(outputs, stop_at_ts=mappings.keys())
    ops_replicate = [op for op in ops if op.type not in NON_REPLICABLE]
    sgv = ge.make_view(*ops_replicate)
    _, info = ge.copy_with_input_replacements(sgv, mappings,
                                              dst_scope=clone_scope)
    return info.transformed(outputs)

def read_graph(path, x):
    is_pytorch = path[-3:] != '.tf'
    model, is_conv, means, stds = eran.read_net_file.read_net(path, x.shape, is_pytorch,
                                                              x)
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
    logits = graph.get_tensor_by_name(output_names[0] + ':0')
    preds = tf.nn.softmax(logits=logits)
    return preds, is_conv, means, stds

def read_eran_model(path, batch_size, num_pixels):
    print(batch_size, num_pixels)
    x = tf.placeholder(tf.float64, shape=(1, num_pixels))

    with tf.variable_scope('root_model', reuse=tf.AUTO_REUSE):
        preds, is_conv, means, stds = read_graph(path, x)

    x_batch = tf.placeholder(tf.float64, shape=(batch_size, num_pixels))
    individual_xs = tf.split(x_batch, batch_size, 0)

    outputs = []
    for clone_id, temp_x in enumerate(individual_xs):
        temp_preds = clone_subgraph([preds], {x: temp_x},
                                    'clone_{}'.format(clone_id))[0]
        outputs.append(temp_preds)
    preds_batch = tf.concat(outputs, 0)

    info = (is_conv, means, stds)
    return x_batch, preds_batch, info, x, preds


def main():
    path = sys.argv[1]
    is_pytorch = path[-3:] != '.tf'

    batch_size = 100
    num_pixels = 784
    nb_classes = 10

    x_batch, preds_batch = read_eran_model(path, batch_size, num_pixels)

    #x = tf.placeholder(tf.float64, shape=(1, num_pixels))

    #with tf.variable_scope('root_model', reuse=tf.AUTO_REUSE):
    #    preds, is_conv, means, stds = read_graph(path, x)

    #x_batch = tf.placeholder(tf.float64, shape=(batch_size, num_pixels))
    #individual_xs = tf.split(x_batch, batch_size, 0)

    #outputs = []
    #for clone_id, temp_x in enumerate(individual_xs):
    #    temp_preds = clone_subgraph([preds], {x: temp_x},
    #                                'clone_{}'.format(clone_id))[0]
    #    outputs.append(temp_preds)
    #preds_batch = tf.concat(outputs, 0)

    sess = tf.Session()
    summary = tf.summary.FileWriter("/tmp/mylog/", sess.graph)

    # get x_test and y_test
    benchmark_tests = eran_benchmark.load_tests('mnist')
    x_test = []
    y_test = []
    for test in benchmark_tests:
        image = np.float64(np.float64(test[1:])/np.float64(255))
        if is_pytorch:
            normalize('mnist', image, means, stds, is_conv)
        x_test.append(image)
        tmp = [0]*nb_classes
        tmp[int(test[0])] = 1
        y_test.append(np.float64(tmp))
    x_test = np.float64(x_test)
    y_test = np.float64(y_test)

    mypreds = sess.run(preds_batch, feed_dict={x_batch: x_test[:batch_size]})
    print(mypreds)

if __name__ == "__main__":
    main()
