import argparse
import logging
import os
import time

import definitions

parser = argparse.ArgumentParser()
parser.add_argument('--netname', type=str, default='../eran_benchmark/nets/pytorch/mnist/convBigRELU__DiffAI.pyt',
                    help='the network name, extensions can be .pyt, .tf, .meta '
                    'for ERAN benchmark and otherwise we accept PyTorch and '
                    'TensorFlow saved models. If the following strings are '
                    'provided, we use the pretrained models in Keras: ' + \
                    ','.join(definitions.PRETRAINED_NETS))
parser.add_argument('--dataset', type=str, default='mnist', help='can be either '
                    'mnist, cifar10 or acasxu for ERAN. For pretrained models, '
                    'it is always ImageNet. For pretrained models retrained on '
                    'smaller datasets select the appropriate one (mnist/cifar10).')
parser.add_argument('--img_epsilon', type=float, default=0.1, help='the epsilon for L_inf perturbation')
parser.add_argument('--thresh', type=float, default=0.2, help='threshold for robustness level')
parser.add_argument('--eta', type=float, default=0.001,
                    help='if more than eta, then return No. Between thresh '
                    'and thresh + eta is the ignorance region.')
parser.add_argument('--delta', type=float, default=0.01, help='confidence at least 1-delta ')
parser.add_argument('--timeout', type=int, default=600, help='timeout in seconds')
parser.add_argument('--alpha', type=float, default=0.4, help='interval size')
parser.add_argument('--pixeldp_bound', action='store_true', help='check for the ' + \
                    'robustness bound that PixelDP guarantees')
parser.add_argument('--batch_mode', action='store_false', help='Enable batch mode for the model'
                    ' inference phase')
parser.add_argument('--single_idx', type=int, default=0, help='single idx eval')
parser.add_argument('--batch_size', type=int, default=1, help='eran batch size')
parser.add_argument('--distance_type', default="linf", help="gpu distance type")

args = parser.parse_args()

print('args: {}'.format(args))
print('netname {} eta {} dataset {} thresh {} delta {} alpha {} timeout {} '
      'img_epsilon {} pixeldp_bound {}'.format(args.netname, args.eta, args.dataset,
                           args.thresh, args.delta, args.alpha, args.timeout,
                           args.img_epsilon, args.pixeldp_bound))

logger = logging.getLogger(__name__)
timestr = time.strftime("%Y_%m_%d-%H:%M:%S")
if not os.path.exists(definitions.LOGS_PATH):
    os.mkdir(definitions.LOGS_PATH)

model_name = os.path.splitext(os.path.basename(args.netname))[0]
logging.basicConfig(filename=os.path.join(definitions.LOGS_PATH,
                                          '{}-eta_{}-delta_{}-thresh_{}-img_eps_{}-{}.log'.format(model_name,
                                                                                                  args.eta,
                                                                                                  args.delta,
                                                                                                  args.thresh,
                                                                                                  args.img_epsilon,
                                                                                                  timestr)),
                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

logger.info('-' * 80)
logger.info(args)
logger.info('-' * 80)

import eran_benchmark
import pretrained_benchmark
import pixeldp_benchmark
import provero

if args.netname in definitions.PIXELDP_NETS:
    pixeldp_benchmark.check_pixeldp(args)
elif args.netname in definitions.PRETRAINED_NETS:
    #check_pytorch_pretrained_model(args)
    pretrained_benchmark.check_tf_pretrained_model(args)
else:
    extension = os.path.splitext(args.netname)[1]
    print(extension)
    # Check for the file extension of the netname -- .pyt and .tf are the ERAN
    # benchmark models so have to use ERAN code to load them. Otherwise, will
    # try to load PyTorch or TensorFlow
    if extension == '.pyt' or extension == '.tf':
        # XXX wanted to add batch input but have to hack into the read_net_file
        # to change the hardcoded dimensions
        eran_benchmark.check_eran_model(args.dataset, args.netname,
                                        args.batch_size, args.img_epsilon, args.thresh,
                                        args.eta, args.delta, args.alpha,
                                        args.timeout, args.batch_mode, args.single_idx,
                                        args.distance_type)
    elif extension == '.pth':
        is_pytorch_model = True
        print('.pth not supported yet')
        exit(1)
    elif extension == '.pb' or extension == '.meta':
        is_tensorflow_model = True
        pretrained_benchmark.check_pixeldp(args)
    else:
        print('Unsupported file/netname {}'.format(args.netname))
        exit(1)

