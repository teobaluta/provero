import os

LOGS_PATH=os.path.join(os.environ['PROVERO_HOME'], 'logs')
ERAN_TEST_PATH=os.path.join(os.environ['PROVERO_HOME'], 'eran_benchmark')
PROVERO_TEST_PATH=os.path.join(os.environ['PROVERO_HOME'], 'provero_benchmark')
NB_TEST_SAMPLES=100

PIXELDP_NETS = ['pixeldp_resnet']
PRETRAINED_NETS =['VGG16', 'VGG19', 'ResNet18', 'ResNet50', 'AlexNet',
                  'DenseNet', 'Inception_v3'] 


