#!/usr/bin/env python


import tensorflow as tf
from keras.applications import vgg16;
from keras.applications import vgg19;
from keras.applications import resnet50;
from keras.applications import densenet;
from keras.applications import inception_v3;
from keras import backend as K
from keras.preprocessing import image
import os
import pickle
#import matplotlib.pyplot as plt
import numpy as np


def _maybe_save(name, model):
    ckpt_dir = os.path.join(os.environ['PROVERO_MODELS_PATH'], name)

    if not os.path.exists(ckpt_dir):
        sess = K.get_session()
        saver = tf.compat.v1.train.Saver()
        os.mkdir(ckpt_dir)
        saver.save(sess, os.path.join(ckpt_dir, name))
        saver.export_meta_graph(os.path.join(ckpt_dir, name, name + '.meta'))

    return (ckpt_dir, model.input, model.output)

def keras_pretrained_model(netname):
    #import pdb; pdb.set_trace()
    if netname == 'VGG16':
        model = vgg16.VGG16(weights='imagenet')
    elif netname == 'VGG19':
        model = vgg19.VGG19(weights='imagenet')
    elif netname == 'ResNet50':
        model = resnet50.ResNet50(weights='imagenet')
    elif netname == 'DenseNet':
        model = densenet.DenseNet121(weights='imagenet')
    elif netname == 'Inception_v3':
        model = inception_v3.InceptionV3(weights='imagenet')

    return _maybe_save(netname, model)

def keras_preprocess_input(netname):
    if netname == 'VGG16':
        return vgg16.preprocess_input
    if netname == 'VGG19':
        return vgg19.preprocess_input
    if netname == 'ResNet50':
        return resnet50.preprocess_input
    if netname == 'DenseNet':
        return densenet.preprocess_input
    if netname == 'Inception_v3':
        return inception_v3.preprocess_input


def validate(model):
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #K.set_session(tf.Session(config=config))

    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.imagenet_utils import decode_predictions

    data_generator = ImageDataGenerator()
    # always take same shuffled order
    seed = 1
    nb_test_samples = 1000
    val_path = os.path.join(os.environ['DATA_DIR'], 'val')
    test_generator = data_generator.flow_from_directory(val_path,
                                                        target_size=(224, 224), seed=seed,
                                                        batch_size=32)

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers:
        print(layer, layer.trainable)

    # Compile model; just for testing
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    loss, acc = model.evaluate_generator(test_generator, val_samples=nb_test_samples)

    print('Loss: {}, Accuracy: {}'.format(loss, acc))

def get_test_images(nb_samples, netname, test_path, debug=False):
    """
    Return a list of samples from the val set
    """
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #K.set_session(tf.Session(config=config))

    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.imagenet_utils import decode_predictions

    # First try to load the images from the local .csv
    # If it is not found we save the images in csv format
    print("Trying to load from: {}".format(test_path))
    if netname == 'Inception_v3':
        tests_pkl = os.path.join(test_path, 'imagenet_test_inception.pkl')
    else:
        tests_pkl = os.path.join(test_path, 'imagenet_test.pkl')

    tests = []
    if os.path.exists(tests_pkl):
        print('.pkl with files exists, loading from it')
        with open(tests_pkl, 'rb') as testsf:
            tests = pickle.load(testsf)
        return tests

    data_generator = ImageDataGenerator()
    # always take same shuffled order
    seed = 3
    val_path = os.path.join(os.environ['DATA_DIR'], 'val')
    if netname == 'Inception_v3':
        test_generator = data_generator.flow_from_directory(val_path,
                                                            target_size=(299, 299),
                                                            seed=seed,
                                                            batch_size=32)
    else:
        test_generator = data_generator.flow_from_directory(val_path,
                                                            target_size=(224, 224),
                                                            seed=seed,
                                                            batch_size=32)

    #test = test_generator.next()
    #image = test[0]
    #correct_label = test[1]

    #print(image[0])
    #print(image[0].shape)

    #img = img_to_array(image[0])
    ##plt.imshow(np.uint8(img))
    ##plt.show()
    #img_batch = np.expand_dims(img, axis=0)
    #imgplot = plt.imshow(np.uint8(img_batch[0]))

    #processed_image = vgg16.preprocess_input(img_batch.copy())
    #predictions = model.predict(processed_image)
    #label = decode_predictions(predictions)
    #print('Predicted label: {}'.format(label))
    #print('Ground truth label: {}'.format(decode_predictions(correct_label)[0]))

    i = 0
    for test in test_generator:
        #print('test batch {}; img {}'.format(test[0].shape, test[0][0].shape))
        #print('correct batch {}'.format(i, test[1].shape))
        img_batch = np.array([img_to_array(x) for x in test[0]])

        #correct_labels = decode_predictions(test[1])
        # correct label clabel is the same as top1 predicted label plabel
        idx = 0
        for clabel in test[1]:
            label = np.argmax(clabel)
            if i >= nb_samples:
                print('Wrote {} samples to {}'.format(nb_samples, tests_pkl))
                with open(tests_pkl, 'wb') as testsf:
                    pickle.dump(tests, testsf)
                return tests
            tests.append([label, img_batch[idx]])
            print('added to test set image {} label {}'.format(i, label))
            idx += 1
            i += 1

