import numpy as np
import time
import logging
import os
import csv

import definitions
import model_loader
import provero

import clone_graph

logger = logging.getLogger(__name__)

def load_tests(dataset):
    if dataset == 'mnist':
        csvfile = open(os.path.join(definitions.ERAN_TEST_PATH, 'mnist_test.csv'), 'r')
        tests = csv.reader(csvfile, delimiter=',')
    elif dataset == 'cifar10':
        csvfile = open(os.path.join(definitions.ERAN_TEST_PATH, 'cifar10_test.csv'), 'r')
        tests = csv.reader(csvfile, delimiter=',')
    else:
        print('dataset not recognized')
        exit(1)

    return tests

# CODE from ERAN, if it is trained with pytorch, we need to normalize the image
# before inference
def normalize(dataset, image, means, stds, is_conv):
    if dataset == 'mnist':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif dataset == 'cifar10':
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        if is_conv:
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1

def denormalize(dataset, image, means, stds, is_conv):
    if dataset == 'mnist':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif dataset == 'cifar10':
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        if is_conv:
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1

def check_eran_model(dataset, netname, batch_size, img_epsilon, thresh, eta,
                     delta, alpha, timeout, batch_mode, single_idx, distance_type):
    """
    Check ERAN benchmark's models
    """
    benchmark_tests = load_tests(dataset)
    nn, load_params = model_loader.load_nn(netname, dataset, batch_size,
                                           batch_mode, distance_type)

    print("mean={},stds={}".format(load_params.means, load_params.stds))

    i = 0
    total_imgs = 0
    verified_imgs = 0
    avg_time = 0.0

    if single_idx:
        i = single_idx

    tests = list(benchmark_tests)

    if single_idx:
        tests = [tests[single_idx-1]]

    for test in tests:
        if dataset == 'mnist':
            image = np.float64(test[1:len(test)])/np.float64(255)

            correct_label = int(test[0])

            # if trained with pytorch should normalize before inference
            if load_params.is_trained_with_pytorch:
                norm_image = np.copy(image)
                normalize(dataset, norm_image, load_params.means, load_params.stds,
                          load_params.is_conv)
                label = nn.infer([norm_image])
            else:
                label = nn.infer([image])

            if label == int(test[0]):
                logger.info('-' * 80)
                logger.info('img {} - label {}'.format(i, label))

                start = time.time()

                f = lambda img: normalize(dataset, img, load_params.means, load_params.stds,
                                          load_params.is_conv)
                if not load_params.is_trained_with_pytorch:
                    f = lambda img: img
                # for pytorch models the robustness we do it on the non-normalized image
                X = provero.RandomVarRobustness(nn, image, f, eps=img_epsilon,
                                                mode='non', distance='linf_eran')
                print("Running adaptive_assert!")
                answer = provero.adaptive_assert(thresh, eta, delta, X, timeout, alpha)

                ans_time = time.time() - start
                avg_time += ans_time
                total_imgs += 1
                if answer == 'Yes':
                    verified_imgs += 1
                    logger.info('#adversarial < {}: {} ({} sec)'.format(thresh, answer, ans_time))
                    print('img {} #adversarial < {}: {} {} sec'.format(i, thresh,
                                                                       answer,
                                                                       ans_time),flush=True)
                    logger.info('-' * 80)
                elif answer is None:
                    print('img {} None {}'.format(i, ans_time), flush=True)
                elif answer is 'timeout':
                    print('img {} timeout {}'.format(i, ans_time), flush=True)
                elif answer == 'No':
                    logger.info('#adversarial < {}: {} ({} sec)'.format(thresh, answer, ans_time))
                    print('img {} #adversarial < {}: {} {} sec'.format(i, thresh,
                                                                       answer,
                                                                       ans_time),
                          flush=True)
                    logger.info('-' * 80)
            else:
                logger.info('SKIP img {} - incorrectly classified as {} instead of {}'.format(i,
                                                                                              label, int(test[0])))
                print('SKIP img {} - incorrectly classified as {} instead of {}'.format(i, label,
                                                                                        int(test[0])), flush=True)
            i += 1


    print('Not considered: {} imgs'.format(i - total_imgs))
    print('Avg time: {} sec'.format(avg_time / total_imgs))
    print('Verified robustness: {} / {}'.format(verified_imgs, total_imgs))



