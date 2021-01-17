import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn

import time

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

import os

DATA_DIR=os.environ['DATA_DIR']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def create_imagenet_loaders(batch_size=16, num_workers=8):
    """
    From
    https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
    """
#    data_transforms = {
#        'train': transforms.Compose([
#            #transforms.RandomResizedCrop(input_size),
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ]),
#        'val': transforms.Compose([
#            #transforms.Resize(input_size),
#            transforms.CenterCrop(input_size),
#            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ]),
#    }

    #imagenet_data = torchvision.datasets.ImageNet(DATA_DIR, download=True,
    #                                              split='val')


    imagenet_data = torchvision.datasets.ImageNet(DATA_DIR, download=True,
                                                  split='val')


    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=10)


    valdir = os.path.join(DATA_DIR, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return val_loader

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def main():
    """
    For testing purposes -- validate the pre-trained models
    """
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    val_loader = create_imagenet_loaders()

    resnet18 = models.resnet18(pretrained=True)
    resnet18 = torch.nn.DataParallel(resnet18).cuda()
    validate(val_loader, resnet18, criterion)

    alexnet = models.alexnet(pretrained=True)
    alexnet.features = torch.nn.DataParallel(alexnet.features)
    alexnet.cuda()
    validate(val_loader, alexnet, criterion)

    vgg16 = models.vgg16(pretrained=True)
    vgg16.features = torch.nn.DataParallel(vgg16.features)
    vgg16.cuda()
    validate(val_loader, vgg16, criterion)

if __name__ == '__main__':
    main()
