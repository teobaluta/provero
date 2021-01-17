from pixeldp_models import pixeldp_resnet

def module_from_name(name):
    if name == 'pixeldp_resnet':
        return pixeldp_resnet
    else:
        raise ValueError('Model "{}" not supported'.format(name))

def name_from_module(module):
    return module.__name__.split('.')[-1]
