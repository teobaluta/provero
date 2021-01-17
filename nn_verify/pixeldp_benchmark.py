import logging
import time

import definitions
import provero

import tensorflow as tf
import keras
import model_loader
import numpy as np
import os
import six
import json

from pixeldp_models import evaluate, pixeldp_resnet, params, pixeldp
from pixeldp_models.utils import robustness
import datasets.cifar

logger = logging.getLogger(__name__)

#MODELS_DIR='/home/projects/11000744/teo/pixeldp/trained_models/'
MODELS_DIR='/hdd/pixeldp/trained_models'
DATASET='cifar10'
DATA_PATH='/hdd/pixeldp/datasets/'


class PixelDPNet(object):
    def __init__(self, batch_size, eval_data_size, images, labels, dir_name=None, dataset=None):
        image_size     = 32
        n_channels      = 3
        num_classes    = 10
        relu_leakiness = 0.1
        #batch_size = 1
        #n_draws    = 2000
        n_draws    = 25
        self.compute_robustness = True
        L = 0.1

        # from main.py if dataset != mnist and dataset != svhn
        steps_num       = 90000
        # XXX THIS IS MODIFIED, originally 10k all test set of CIFAR10
        #eval_data_size  = 1
        lrn_rate        = 0.1
        lrn_rte_changes = [40000, 60000, 80000]
        lrn_rte_vals    = [0.01, 0.001, 0.0001]

        self.hps = params.HParams(
                name_prefix="",
                batch_size=batch_size,
                num_classes=num_classes,
                image_size=image_size,
                n_channels=n_channels,
                lrn_rate=lrn_rate,
                lrn_rte_changes=lrn_rte_changes,
                lrn_rte_vals=lrn_rte_vals,
                num_residual_units=4,
                use_bottleneck=False,
                weight_decay_rate=0.0002,
                relu_leakiness=relu_leakiness,
                optimizer='mom',
                image_standardization=False,
                n_draws=n_draws,
                dp_epsilon=1.0,
                dp_delta=0.05,
                robustness_confidence_proba=0.05,
                attack_norm_bound=L,
                attack_norm='l2',
                sensitivity_norm='l2',
                sensitivity_control_scheme='bound',  # bound or optimize
                noise_after_n_layers=1,
                layer_sensitivity_bounds=['l2_l2'],
                noise_after_activation=True,
                parseval_loops=10,
                parseval_step=0.0003,
                steps_num=steps_num,
                eval_data_size=eval_data_size,
        )

        config = tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.visible_device_list = str(dev.split(":")[-1])
        self.sess = tf.Session(config=config)

        tf.train.start_queue_runners(self.sess)
        self.model = pixeldp_resnet.Model(self.hps, images, labels, 'eval')
        self.model.build_graph()

        if dir_name == None:
            #dir_name = FLAGS.models_dir
            dir_name = MODELS_DIR

        self.dir_name = os.path.join(dir_name,
                                     params.name_from_params(pixeldp_resnet,
                                                             self.hps))
        if dataset == None:
            #dataset = FLAGS.dataset
            dataset = DATASET

        saver = tf.train.Saver()

        self.summary_writer = tf.summary.FileWriter(self.dir_name)

        try:
            print('Loading from {}'.format(self.dir_name))
            ckpt_state = tf.train.get_checkpoint_state(self.dir_name)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(self.sess, ckpt_state.model_checkpoint_path)


    def infer(self, feed_dict=[]):
        """
        Expecting inputs images as first elem, labels as second. If none, input
        is read from the queue (from file input reader)
        """
        args = {}
        if self.model.noise_scale == None:
            args = {}  # For Madry and inception
        else:
            args = {self.model.noise_scale: 1.0}

        if len(feed_dict) > 1:
            args[self.model.images] = feed_dict[0]
            args[self.model.labels] = feed_dict[1]
        start_time = time.time()
        (
            loss,
            softmax_predictions,
            truth,
            self.images
         ) = self.sess.run(
            [
                self.model.cost,
                self.model.predictions,
                self.model.labels,
                self.model.images,
            ],
             feed_dict=args)
        #logger.debug('session run on feed_dict = {}'.format(time.time() - start_time))

        self.labels = truth
        truth = np.argmax(truth, axis=1)[:self.hps.batch_size]
        prediction_votes = np.zeros([self.hps.batch_size, self.hps.num_classes])
        softmax_sum = np.zeros([self.hps.batch_size, self.hps.num_classes])
        softmax_sqr_sum = np.zeros([self.hps.batch_size, self.hps.num_classes])
        predictions = np.argmax(softmax_predictions, axis=1)

        return loss, truth, predictions, prediction_votes, softmax_predictions, \
    softmax_sum, softmax_sqr_sum

    def draws_prediction(self, loss, truth, predictions, prediction_votes,
                         softmax_predictions, softmax_sum,
                         softmax_sqr_sum):

        start_time = time.time()
        # Make predictions on the dataset, keep the label distribution
        total_prediction, correct_prediction_argmax, correct_prediction_logits = 0, 0, 0
        data = {
                'argmax_sum': [],
                'softmax_sum': [],
                'softmax_sqr_sum': [],
                'pred_truth_argmax':  [],
                'pred_truth_softmax':  [],
                }

        #logger.debug('Doing {} draws for prediction'.format(self.hps.n_draws))
        for i in range(self.hps.n_draws):
            for j in range(self.hps.batch_size):
                prediction_votes[j, predictions[i*self.hps.batch_size + j]] += 1
                softmax_sum[j] += softmax_predictions[i*self.hps.batch_size + j]
                softmax_sqr_sum[j] += np.square(
                        softmax_predictions[i*self.hps.batch_size + j])

        predictions = np.argmax(prediction_votes, axis=1)
        #logger.debug('Predictions after {} draws {} took {} '
        #             'sec'.format(self.hps.n_draws, predictions, time.time() -
        #                          start_time))
        predictions_logits = np.argmax(softmax_sum, axis=1)

        data['argmax_sum'] += prediction_votes.tolist()
        data['softmax_sum'] += softmax_sum.tolist()
        data['softmax_sqr_sum'] += softmax_sqr_sum.tolist()
        data['pred_truth_argmax']  += (truth == predictions).tolist()
        data['pred_truth_softmax'] += (truth == predictions_logits).tolist()

        #logger.debug("From argamx: {} / {}".format(np.sum(truth == predictions), len(predictions)))
        #logger.debug("From logits: {} / {}".format(np.sum(truth == predictions_logits), len(predictions)))

        return data

    def pixeldp_infer(self, feed_dict=[]):
        if len(feed_dict) > 1:
            image = feed_dict[0]
            label = feed_dict[1]
            loss, truth, predictions, prediction_votes, softmax_predictions, \
                softmax_sum, softmax_sqr_sum = self.infer([image, label])
        else:
            loss, truth, predictions, prediction_votes, softmax_predictions, \
                softmax_sum, softmax_sqr_sum = self.infer()
        #logger.debug('Fwd pass done. Sampling from last layer softmax')
        data = self.draws_prediction(loss, truth, predictions, prediction_votes,
                                     softmax_predictions, softmax_sum,
                                     softmax_sqr_sum)
        predictions = np.argmax(data['argmax_sum'], axis=1)
        #print('Pred truth argmax {}'.format(data['pred_truth_argmax']))
        #print('Pred truth softmax {}'.format(data['pred_truth_softmax']))
        #print('Truth {}; Prediction {}'.format(truth, prediction))
        # logging the loss
        self.loss = loss

        return truth, predictions, data

def check_pixeldp(args, batch_size=1):
    """
    Very hackish way of instantiating the PixelDP model
    """
    image_size = 32
    nb_channels = 3

    total_prediction, correct_prediction_argmax, correct_prediction_logits = 0, 0, 0
    eval_batch_size = 1
    eval_data_size = 24
    eval_batch_count = int(eval_data_size/eval_batch_size)


    avg_time = 0
    total_imgs = 0
    verified_imgs = 0
    images, labels = datasets.cifar.build_tests(
            DATASET,
            DATA_PATH,
            batch_size,
            False
        )
    print('Loading PixelDP model, evaluate on 100 images.')
    nn = PixelDPNet(batch_size=1, eval_data_size=100, images=images, labels=labels)

    image = np.zeros([batch_size, image_size, image_size, nb_channels],
                     dtype=np.float32)
    label = np.zeros([batch_size, 10])
    label[0] = 1

    for i in six.moves.range(eval_batch_count):
        start_time = time.time()
        print('Infer on images {} with labels {}'.format(images, labels))

        truth, predictions, data = nn.pixeldp_infer()
        print("Done: {}/{}".format(eval_batch_size*i, eval_data_size))
        print('Prediction time on image from {} = {}'.format(DATASET,
                                                             time.time() -
                                                             start_time))

        predictions = np.argmax(data['argmax_sum'], axis=1)
        predictions_logits = np.argmax(data['softmax_sum'], axis=1)
        correct_prediction_argmax += np.sum(truth == predictions)
        correct_prediction_logits += np.sum(truth == predictions_logits)
        total_prediction   += predictions.shape[0]

        current_precision_argmax = 1.0 * correct_prediction_argmax / total_prediction
        current_precision_logits = 1.0 * correct_prediction_logits / total_prediction
        #logger.debug("Current precision from argmax: {}".format(current_precision_argmax))
        #logger.debug("Current precision from logits: {}".format(current_precision_logits))
        # For Parseval, get true sensitivity, use to rescale the actual attack
        # bound as the nosie assumes this to be 1 but often it is not.
        # Parseval updates usually have a sensitivity higher than 1
        # despite the projection: we need to rescale when computing
        # sensitivity.
        start_time = time.time()
        if nn.model.pre_noise_sensitivity() == None:
            sensitivity_multiplier = None
        else:
            sensitivity_multiplier = float(nn.sess.run(
                nn.model.pre_noise_sensitivity(),
                {nn.model.noise_scale: 1.0}
            ))

        #logger.debug('Parseval; get true sensitivity = {} s'.format(time.time()
        #                                                            - start_time))
        with open(nn.dir_name + "/sensitivity_multiplier.json", 'w') as f:
            d = [sensitivity_multiplier]
            f.write(json.dumps(d))

        # Compute robustness and add it to the eval data.
        if nn.compute_robustness:  # This is used mostly to avoid errors on non pixeldp DNNs
            dp_mechs = {
                'l2': 'gaussian',
                'l1': 'laplace',
            }
            robustness_from_argmax = [robustness.robustness_size_argmax(
                counts=x,
                eta=nn.hps.robustness_confidence_proba,
                dp_attack_size=nn.hps.attack_norm_bound,
                dp_epsilon=nn.hps.dp_epsilon,
                dp_delta=nn.hps.dp_delta,
                dp_mechanism=dp_mechs[nn.hps.sensitivity_norm]
                ) / sensitivity_multiplier for x in data['argmax_sum']]
            data['robustness_from_argmax'] = robustness_from_argmax
            robustness_from_softmax = [robustness.robustness_size_softmax(
                tot_sum=data['softmax_sum'][i],
                sqr_sum=data['softmax_sqr_sum'][i],
                counts=data['argmax_sum'][i],
                eta=nn.hps.robustness_confidence_proba,
                dp_attack_size=nn.hps.attack_norm_bound,
                dp_epsilon=nn.hps.dp_epsilon,
                dp_delta=nn.hps.dp_delta,
                dp_mechanism=dp_mechs[nn.hps.sensitivity_norm]
                ) / sensitivity_multiplier for i in range(len(data['argmax_sum']))]
            data['robustness_from_softmax'] = robustness_from_softmax

        data['sensitivity_mult_used'] = sensitivity_multiplier

        eval_data_name = '/eval_data-eta_{}'.format(args.eta) + \
            '-delta_{}-thresh_{}'.format(args.delta, args.thresh) + \
            '-img_eps_{}-img_{}.json'.format(args.img_epsilon, i)
        # Log eval data
        with open(nn.dir_name + eval_data_name, 'w') as f:
            f.write(json.dumps(data))

        # Print stuff
        precision_argmax = 1.0 * correct_prediction_argmax / total_prediction
        precision_logits = 1.0 * correct_prediction_logits / total_prediction

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision argmax', simple_value=precision_argmax)
        precision_summ.value.add(
            tag='Precision logits', simple_value=precision_logits)
        #summary_writer.add_summary(precision_summ, train_step)
        #  summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision argmax: %.3f, precision logits: %.3f' %
                        (nn.loss, precision_argmax, precision_logits))
        nn.summary_writer.flush()
        #logger.debug(data)

        print(data)
        if predictions == truth:
            print('-' * 80)
            logger.debug('-' * 80)
            print('img {} - label {}'.format(i, truth))
            logger.debug('img {} - label {}'.format(i, truth))
            start = time.time()

            preprocess = lambda img: img
            print('Prediction on {} images'.format(len(nn.images)))
            assert(len(nn.images.shape) == 4)
            assert(nn.images.shape[0] == 1)
            if args.pixeldp_bound is True:
                robustness_from_argmax = data['robustness_from_argmax']
                print('PixelDP robustness_from_argmax = {}'.format(robustness_from_argmax))
                print('Running provero with img_epsilon = {}'.format(robustness_from_argmax))
                X = verro.RandomVarRobustness(nn, nn.images[0], preprocess,
                                              eps=robustness_from_argmax,
                                              mode='non', distance='l2',
                                              pixeldp=[nn.images, nn.labels])

                answer = verro.adaptive_assert(args.thresh, args.eta, args.delta,
                                               X, args.timeout, args.alpha)

                print('PixelDP robustness_from_softmax = {}'.format(data['robustness_from_softmax']))
                robustness_from_argmax = data['robustness_from_softmax']
                print('PixelDP robustness_from_argmax = {}'.format(robustness_from_softmax))
                print('Running provero with img_epsilon = {}'.format(robustness_from_softmax))
                X = verro.RandomVarRobustness(nn, nn.images[0], preprocess,
                                              eps=robustness_from_softmax,
                                              mode='non', distance='l2',
                                              pixeldp=[nn.images, nn.labels])

                answer = verro.adaptive_assert(args.thresh, args.eta, args.delta,
                                               X, args.timeout, args.alpha)
            else:
                X = verro.RandomVarRobustness(nn, nn.images[0], preprocess,
                                              eps=args.img_epsilon,
                                              mode='non', distance='l2',
                                              pixeldp=[nn.images, nn.labels])
                answer = verro.adaptive_assert(args.thresh, args.eta, args.delta,
                                               X, args.timeout, args.alpha)


            ans_time = time.time() - start
            avg_time += ans_time
            total_imgs += 1
            if answer == 'Yes':
                verified_imgs += 1

            if answer is None:
                print('img {} timeout {}'.format(i, ans_time), flush=True)
                i += 1
                continue

            logger.info('#adversarial < {}: {} ({} sec)'.format(args.thresh, answer, ans_time))
            print('img {} #adversarial < {}: {} {} sec'.format(i, args.thresh, answer, ans_time), flush=True)
            logger.info('-' * 80)
        else:
            logger.info('SKIP img {} - incorrectly classified as {} instead of'
                        ' {}'.format(i, predictions, truth))
            print('SKIP img {} - incorrectly classified as {} instead of'
                  ' {}'.format(i, predictions, truth), flush=True)

    print('Not considered: {} imgs'.format(i - total_imgs))
    if total_imgs != 0:
        print('Avg time: {} sec'.format(avg_time / total_imgs))
    print('Verified robustness: {} / {}'.format(verified_imgs, total_imgs))


