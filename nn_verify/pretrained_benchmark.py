from pretrained_models import pretrained_tf
import logging
import time

import definitions
import provero

import tensorflow as tf
import keras
import model_loader

logger = logging.getLogger(__name__)

def check_tf_pretrained_model(args):
    """
    Use this class to load pretrained models in Keras.
    """

    #nn_input, nn_output = pretrained_tf.keras_pretrained_model(args.netname)
    #print(nn.input)
    #print(nn.output)

    #print(nn.summary())
    # Load ImageNet and validate model on testset
    #pretrained_tf.validate(model)

    nn, _ = model_loader.load_nn(args.netname, args.dataset, args.batch_size)
    tests = pretrained_tf.get_test_images(definitions.NB_TEST_SAMPLES, 
                                          args.netname, definitions.PROVERO_TEST_PATH)

    i = 0
    avg_time = 0
    total_imgs = 0
    verified_imgs = 0

    if args.single_idx:
        tests = [tests[args.single_idx-1]]

    for test in tests:
        correct_label = test[0]
        image = test[1]
        # NOTE that we assume the tests are correctly classified images in
        # ImageNet

        print(args.netname)
        preprocess = pretrained_tf.keras_preprocess_input(args.netname)
        import numpy as np
        preprocess_image = np.copy(image)
        preprocess_image = preprocess(preprocess_image)
        label = nn.infer([preprocess_image])[0]
        if label == correct_label:
            logger.debug('-' * 80)
            logger.debug('img {} - label {}'.format(i, correct_label))

            img_epsilon = args.img_epsilon
            start = time.time()

            X = provero.RandomVarRobustness(nn, image, preprocess, eps=img_epsilon)
            answer = provero.adaptive_assert(args.thresh, args.eta, args.delta, X, args.timeout,
                                             args.alpha)

            ans_time = time.time() - start
            avg_time += ans_time
            total_imgs += 1
            if answer == 'Yes':
                verified_imgs += 1
                logger.info('#adversarial < {}: {} ({} sec)'.format(args.thresh, answer, ans_time))
                print('img {} #adversarial < {}: {} {} sec'.format(i, args.thresh, answer, ans_time), flush=True)
                logger.info('-' * 80)
            elif answer == 'No':
                logger.info('#adversarial < {}: {} ({} sec)'.format(args.thresh, answer, ans_time))
                print('img {} #adversarial < {}: {} {} sec'.format(i, args.thresh, answer, ans_time), flush=True)
                logger.info('-' * 80)
            elif answer is None:
                print('img {} None {}'.format(i, ans_time), flush=True)
            elif answer is 'timeout':
                print('img {} timeout {}'.format(i, ans_time), flush=True)

        else:
            logger.info('SKIP img {} - incorrectly classified as {} instead of {}'.format(i, label, correct_label))
            print('SKIP img {} - incorrectly classified as {} instead of {}'.format(i, label, correct_label), flush=True)
        i += 1

    print('Not considered: {} imgs'.format(i - total_imgs))
    print('Avg time: {} sec'.format(avg_time / total_imgs))
    print('Verified robustness: {} / {}'.format(verified_imgs, total_imgs))


