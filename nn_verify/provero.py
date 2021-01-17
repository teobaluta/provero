import math
import numpy as np
import time
import logging
import time

logger = logging.getLogger(__name__)

class RandomVar(object):
    def __init__(self, mu, sigma):
        print("Init random var")
        self.mu, self.sigma =  mu, sigma

    def sample(self, number_samples, timeout):
        """
        timeout is a dummy parameter for this method
        """
        print(self.mu, self.sigma)
        s = np.random.normal(self.mu, self.sigma, number_samples)
        return s

class RandomVarRobustness(object):
    '''
    explanation of different metrics https://arxiv.org/pdf/1608.04644.pdf
    '''
    def __init__(self, nn, image, normalize, eps=0.1, mode='rgb',
                 distance='linf', pixeldp=[]):
        logger.debug("Init random var for local robustness")
        self.image = image
        self.nn = nn
        # scale the images (VGG16 images are RGB)
        if mode == 'rgb':
            #self.eps = eps * 255
            self.eps = eps
        elif mode == 'non':
            self.eps = eps
        else:
            print('Unknown mode for images {}'.format(mode))
            exit(1)
        self.distance = distance
        self.norm_image = np.copy(image)
        if distance == 'linf_eran':
            normalize(self.norm_image)
        else:
            self.norm_image = normalize(self.norm_image)
        self.normalize = normalize
        self.pixeldp = pixeldp
        if len(pixeldp) > 1:
            self.correct_label, _, _ = nn.pixeldp_infer(pixeldp)
        else:
            self.correct_label = nn.infer([self.norm_image])[0]
        print('RandomVar initialized - ' +
              'correct_label={}; '.format(self.correct_label) +
              'eps={}'.format(self.eps))

    def sample(self, number_samples, timeout):
        """
        Sample uniformly from the sample space of the random variable. For
        robustness, sample from the Lp ball where p=0,1,2,infinity and return
        the number of adversarial samples.
        Mean can be computed as the number of adversarial samples /
        number_samples.

        Given the image (self.image), generate a number_samples images that are
        within self.eps distance in Lp norm and get the predictions of the
        neural net self.nn on them.

        Args:
        number_samples (int): number of samples to draw
        timeout (float): timeout for the sampling
        """
        s = []
        start = time.time()
        if self.distance == 'l0':
            # if distance is L0 - the number of pixels changed
            # chanright ge a number_coord samples in the array
            # XXX this code is probably broken cause of changes!!
            print('l0 distance should check & update the code! exiting')
            exit(1)
            for i in range(1, number_samples):
                #iteration = time.time()
                indices = np.random.choice(self.image.size, math.ceil(self.eps * self.image.size))
                new_x = np.array(self.image, copy=True)
                random_values = np.random.random_sample(indices.size)
                np.put(new_x, indices, random_values)
                #logger.debug('Random indices took {}'.format(time.time() - iteration))
                if self.correct_label == self.nn.infer([new_x]):
                    s.append(0)
                else:
                    s.append(1)

                if i % 9000 == 0:
                    logger.debug('{} sampled so far - time {}'.format(i, time.time() - start))

                s = np.sum(s)
        elif self.distance == 'l1':
            # (for each pixel, calculate the absolute value difference, and
            # sum it over all pixels)
            print('l1 distance not implemented. exiting')
            exit(1)
        elif self.distance == 'l2':
            print('Run for {} samples'.format(number_samples))
            s = 0
            for i in range(0, number_samples):
                norm_x = np.random.normal(0, 1, self.image.shape)
                norm_x = norm_x / np.linalg.norm(norm_x)
                u = np.random.uniform(0, 1, self.image.shape) ** (1 / \
                    (self.image.shape[0] * self.image.shape[1] *
                     self.image.shape[2]))
                new_x = self.image + self.eps * u * norm_x
                assert(np.linalg.norm(new_x - self.image) <= self.eps)
                new_x = np.expand_dims(new_x, axis=0)
                if len(self.pixeldp) > 1:
                    pred_time = time.time()
                    truth, pred, _ = self.nn.pixeldp_infer([new_x,
                                                            self.pixeldp[1]])
                    #logger.debug('PixelDP pred time {}'.format(time.time() - pred_time))
                else:
                    pred = self.nn.infer([new_x])
                if self.correct_label == pred:
                    #s.append(0)
                    s += 0
                else:
                    #s.append(1)
                    s += 1
                if time.time() - start >= timeout:
                    return ['timeout']
        elif self.distance == 'linf':
            # (for each pixel, take the absolute value difference between X
            # and Z, and return the largest such distance found over all
            # pixels)
            # random_eps = np.random.random_sample(self.image.size)
            print('Run for {} samples'.format(number_samples))
            s = 0
            remaining = number_samples
            while remaining > 0:
                zl_samples = []
                zl_bs = self.nn.nn_batch_size
                cur_num = min(remaining, zl_bs)
                remaining -= cur_num
                #assert(remaining >= 0)
                zl_samples = self.nn.sample(self.image, self.eps)
                #for i in range(0, zl_bs):
                #    random_change = np.random.random_sample(self.image.shape)
                #    # scale to -1,1
                #    random_change = random_change * 2 - 1
                #    # scale to epsilon
                #    random_change = random_change * self.eps
                #    zl_samples.append(random_change)
                #zl_samples = zl_samples + self.image

                #print(zl_samples.shape)
                #print(self.image.shape)
                img_diff = zl_samples - self.image
                #assert(np.all(np.less_equal(np.linalg.norm(img_diff, np.inf, 0), self.eps+1e-4)))
                #print(zl_samples[0])
                zl_samples = self.normalize(zl_samples)
                labels = self.nn.infer(zl_samples)[:cur_num]
                #print(labels)
                s += np.sum(np.where(labels == self.correct_label, 0, 1))
                if time.time() - start >= timeout:
                    return ['timeout']
                #print("Remaining: {}".format(remaining))
            print('s:{}'.format(s))
        elif self.distance == 'linf_eran':
            if hasattr(self.nn, 'batch_size'):
                print('Run for {} samples'.format(number_samples))
                s = 0
                remaining = number_samples
                while remaining > 0:
                    zl_samples = []
                    zl_bs = self.nn.batch_size
                    cur_num = min(remaining, zl_bs)
                    remaining -= cur_num
                    zl_samples = self.nn.sample(self.image, self.eps)
                    #for i in range(0, zl_bs):
                    #    random_change = np.random.random_sample(self.image.shape)
                    #    # scale to -1,1
                    #    random_change = random_change * 2 - 1
                    #    # scale to epsilon
                    #    random_change = random_change * self.eps
                    #    zl_samples.append(random_change)
                    #zl_samples = zl_samples + self.image
                    img_diff = zl_samples - self.image
                    #print(np.linalg.norm(img_diff, np.inf, 0))
                    #print(np.less_equal(np.linalg.norm(img_diff, np.inf, 0),
                    #      self.eps+0.0001))
                    #assert(np.all(np.less_equal(np.linalg.norm(img_diff,
                    #                                           np.inf, 0),
                    #                            self.eps+1e-9)))
                    self.normalize(zl_samples)
                    labels = self.nn.infer_batch(zl_samples)[:cur_num]
                    s += np.sum(np.where(labels == self.correct_label, 0, 1))
                    #for i, label in enumerate(labels):
                    #    if label != self.correct_label:
                    #        diff = zl_samples[i] - self.norm_image
                    #        print(np.linalg.norm(diff, np.inf),
                    #              np.linalg.norm(img_diff[i], np.inf))
                    if time.time() - start >= timeout:
                        return ['timeout']
                    #print("Remaining: {}".format(remaining))
                print('s:{}'.format(s))
            else:
                assert(False)
                # Because I didn't have the patience to change to batch inference
                # for the models in the ERAN benchmark
                # NOTE old code without possibility of batch inference
                # I tried on VGG16 thresh 0.01 eps 0.01 takes about 350s / img
                # With batched inference, takes about 100s / img
                print('Run for {} samples'.format(number_samples))
                s = 0
                for i in range(0, number_samples):
                    # change < self.eps
                    random_change = np.random.random_sample(self.image.shape) * self.eps
                    # random direction of change
                    new_x = self.image + random_change * np.random.choice([-1.0, 1.0])
                    self.normalize(new_x)
                    label_x = self.nn.infer([new_x])
                    if isinstance(label_x, list):
                        label_x = label_x[0]
                    if self.correct_label == label_x:
                        #s.append(0)
                        s += 0
                    else:
                        #s.append(1)
                        s += 1
                    if time.time() - start >= timeout:
                        return ['timeout']
                print('s:{}'.format(s))


        else:
            logger.debug('{} not supported'.format(self.distance))
            exit(1)

        return s


class UniformHypersphere(object):
    def __init__(self, d):
        self.d = d

    def sample(self):
        # On decompositional algorithms for uniform sampling from n-spheres and
        # n-balls

        # an array of (d+2) normally distributed random variables
        d = self.d
        u = np.random.normal(0,1,d+2)
        norm=np.sum(u**2) **(0.5)
        u = u/norm
        x = u[0:d] #take the first d coordinates
        return x

###############################################################################
# ADAPTIVE ALGORITHM
###############################################################################
def divide_et_impera(theta, eta, theta1, theta2, left):
    """
    TODO make this uniform interface as fixed_intervals
    """
    if theta == 0 and left:
        return theta, theta + eta
    # XXX we do not support this case so for now we just return
    if theta == 1:
        print('Always true xD!')
        exit(1)
    #if theta == 1 and not left:
    #    return theta - eta, d

    if theta1 == 0 and theta2 == 0 and left:
        return 0, theta

    if theta1 == 0 and theta2 == 0 and not left:
        return theta + eta, 1

    alpha = theta2 - theta1
    if left:
        return theta2 - max(eta, alpha / 2), theta2

    logger.debug('max of eta and alpha/2: {}'.format(max(eta,alpha/2)))
    logger.debug('theta1={}, theta1 + max = {}'.format(theta1,
                                                       theta1+max(eta,alpha/2)))
    return theta1, theta1 + max(eta, alpha / 2)


def adaptive_assert(theta, eta, delta, obj_to_sample, timeout, alpha,
                    interval_strategy=divide_et_impera):
    if obj_to_sample is None:
        return

    theta1_left = 0
    theta2_left = 0
    theta1_right = 0
    theta2_right = 0

    # adjust the delta
    n = 3 + max(0, math.log2(theta / eta)) + max(0, math.log2((1- theta - eta) /
                                                              eta))
    delta = delta / n
    logger.debug('Running with delta={}'.format(delta))

    start_time = time.time()
    while True:
        if time.time() - start_time >= timeout:
            print('timeout!')
            return 'timeout'

        theta1_left, theta2_left = interval_strategy(theta, eta, theta1_left, theta2_left, True)
        assert(theta1_left < theta2_left)
        assert(theta >= theta2_left)
        if theta2_left - theta1_left <= eta:
            logger.debug('left side interval <= eps. test on the right side')
            logger.debug('skip to right side')
        else:
            tester_ans = tester(theta1_left, theta2_left, delta, obj_to_sample,
                                timeout - (time.time() - start_time))

            logger.debug('mean < theta1: tester(alpha={}, theta1_left={}, theta2_left={},'
                         'delta={}) = {}'.format(theta2_left - theta1_left,
                                                 theta1_left, theta2_left,
                                                 delta, tester_ans))
            if tester_ans is 'timeout':
                return 'timeout'

            if tester_ans is 'Yes':
                return 'Yes'

        theta1_right, theta2_right = interval_strategy(theta, eta, theta1_right,
                                                       theta2_right, False)
        logger.debug('alpha = {}, theta2_righ={} after interval_strat'.format(theta2_right-theta1_right,theta2_right))

        assert(theta1_right < theta2_right)
        assert(theta + eta <= theta1_right)
        if theta2_right - theta1_right - eta <= 1e-6:
            logger.debug('right side interval <= eps. test on the left side')
            if theta2_left - theta1_left <= eta:
                logger.debug('left side interval also <= eps. do right-side expensive test')
                tester_ans = tester(theta, theta + eta, delta, obj_to_sample,
                                    timeout - (time.time() - start_time))
                logger.debug('mean < theta1: tester(alpha={}, theta={}, theta+eta={},'
                             'delta={}) = {}'.format(eta, theta, theta + eta,
                                                     delta, tester_ans))
                logger.debug('We just return from here {}'.format(tester_ans))
                return tester_ans
        else:
            tester_ans = tester(theta1_right, theta2_right, delta, obj_to_sample,
                                timeout - (time.time() - start_time))

            logger.debug('mean < theta1: tester(alpha={}, theta1_right={}, theta2_right={},'
                         'delta={}) = {}'.format(theta2_right - theta1_right,
                                                 theta1_right, theta2_right,
                                                 delta, tester_ans))

            if tester_ans is 'No':
                return 'No'

            if tester_ans is 'timeout':
                return 'timeout'


###############################################################################
# NON-ADAPTIVE ALGORITHM
###############################################################################

def check_if_fits(thresh, alpha, eta):
    # check if thresh is smaller than the eta
    if thresh < eta:
        print('thresh {} < eps {}, exiting...'.format(thresh, eta))
        return -1

    if thresh / alpha < 1:
        print('Can\'t use the alpha={} interval size')
        return 0

    return 1

def fixed_intervals(theta, alpha, eta):
    """
    theta - threshold value
    alpha - interval size
    eta - the smallest interval
    """
    intervals = []
    fits = check_if_fits(theta, alpha, eta)
    if fits < 0:
        return []

    if not fits:
        first_interval = (0, theta)
        intervals.append(first_interval)
    else:
        left_no_intervals = math.ceil(theta / alpha)
        # actual alpha left is
        actual_fit = theta / left_no_intervals
        for i in range(left_no_intervals):
            theta1 = i * alpha
            intervals.append((theta1, theta1 + actual_fit))

    fits = check_if_fits(1 - theta - eta, alpha, eta)
    if fits < 0:
        return []
    if not fits:
        last_interval = (theta + eta, 1)
        intervals.append(last_interval)
    else:
        right_no_intervals = math.ceil((1 - theta - eta) / alpha)
        # actual alpha is this
        actual_fit = (1 - theta - eta) / right_no_intervals
        theta1 = theta + eta
        for i in range(right_no_intervals):
            intervals.append((theta1, theta1 + actual_fit))
            theta1 += actual_fit

    intervals.append((theta, theta + eta))

    return intervals, left_no_intervals, right_no_intervals


def non_adaptive_assert(theta, eta, delta, obj_to_sample, timeout, alpha,
                        interval_strategy=fixed_intervals):

    intervals, left_no_intervals, right_no_intervals = interval_strategy(theta, alpha, eta)

    assert(len(intervals) > 0)
    logger.debug('Intervals: {}'.format(intervals))

    delta_min_left = delta / left_no_intervals
    delta_min_right = delta / right_no_intervals

    start_time = time.time()
    for i in range(len(intervals)):
        logger.debug('call tester for [{}], delta={}'.format(intervals[i],
                                                             delta_min_left))

        tester_ans = tester(intervals[i][0], intervals[i][1], delta_min_left,
                            obj_to_sample, timeout - (time.time() - start_time))

        print('tester([{}]) = {}'.format(intervals[i], tester_ans))

        if tester_ans is 'Yes' and theta > intervals[i][1]:
            return 'Yes'

        i_right = len(intervals) - 1 - i
        logger.debug('call tester for [{}], delta={}'.format(intervals[i_right],
                                                             delta_min_right))

        tester_ans = tester(intervals[i_right][0], intervals[i_right][1],
                            delta_min_right, obj_to_sample,
                            timeout - (time.time() - start_time))

        logger.debug('tester([{}]) = {}'.format(intervals[i_right], tester_ans))

        if tester_ans is 'No' and theta < intervals[i_right][0]:
            return 'No'

###############################################################################
# TESTER Primitive
###############################################################################

def tester(theta1, theta2, delta, obj_to_sample, timeout):
    logger.debug('[tester] alpha = {}, delta = {}, '
                 'theta1 = {}, theta2={}'.format(theta2 - theta1, delta, theta1, theta2))
    assert(theta2 > theta1)
    if theta2 > 1:
        theta2 = 1
    if theta1 < 0:
        theta1 = 0

    # compute number of samples
    N = 1/((theta2 - theta1)**2) * math.log(1/(delta))*(math.sqrt(3*theta1)+math.sqrt(2*(theta2)))**2
    N = math.ceil(N)

    logger.debug("Sample {} times".format(N))
    start = time.time()
    s = obj_to_sample.sample(N, timeout)
    logger.debug("s = {}".format(s))

    if isinstance(s, str):
        if s == 'timeout':
            return 'timeout'

    if isinstance(s, list):
        if s[0] == 'timeout':
            return 'timeout'

    logger.debug("Sampling took {} sec".format(time.time() - start))
    print("Sampling took {} sec".format(time.time() - start))
    #mean = np.mean(s)
    #import pdb; pdb.set_trace()
    mean = s / N
    logger.debug("mean = {}".format(mean))
    if mean <= theta1:
        return 'Yes'

    if mean > theta2:
        return 'No'

