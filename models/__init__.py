import models.mnist
import models.cifar
import models.imagenet


def make_model(dataset, arch, **kwargs):
    """
    Make model, and load pre-trained weights.
    :param dataset: mnist, cifar10, cifar100, svhn, or imagenet
    :param arch: arch name, e.g., alexnet
    :return: model (in cpu and training mode)
    """
    assert dataset in ['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet']
    if dataset == 'mnist':
        if arch == 'mlp':
            model = models.mnist.mlp(**kwargs)
        elif arch == 'mlp800':
            model = models.mnist.mlp800(**kwargs)
        elif arch == 'lenet':
            model = models.mnist.lenet(**kwargs)
        elif arch == 'liunet':
            model = models.mnist.liunet(**kwargs)
        else:
            raise NotImplementedError('Unknown arch {} for dataset {}'.format(arch, dataset))
    elif dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'svhn':
        if arch == 'lenet':
            model = models.cifar.lenet(**kwargs)
        elif arch == 'nin':
            model = models.cifar.nin(**kwargs)
        elif arch == 'liunet64':
            model = models.cifar.liunet(width=64, **kwargs)
        elif arch == 'liunet96':
            model = models.cifar.liunet(width=96, **kwargs)
        elif arch == 'alexnet':
            model = models.cifar.alexnet(**kwargs)
        elif arch == 'alexnet_bn':
            model = models.cifar.alexnet_bn(**kwargs)
        elif arch == 'vgg11_bn':
            model = models.cifar.vgg11_bn(**kwargs)
        elif arch == 'vgg13_bn':
            model = models.cifar.vgg13_bn(**kwargs)
        elif arch == 'vgg16_bn':
            model = models.cifar.vgg16_bn(**kwargs)
        elif arch == 'vgg19_bn':
            model = models.cifar.vgg19_bn(**kwargs)
        elif arch == 'densenet40':
            model = models.cifar.densenet(depth=40, growthRate=12, compressionRate=1, **kwargs)
        elif arch == 'densenet100':
            model = models.cifar.densenet(depth=100, growthRate=12, **kwargs)
        elif arch == 'densenet190':
            model = models.cifar.densenet(depth=190, growthRate=40, **kwargs)
        elif arch == 'preresnet110':
            model = models.cifar.preresnet(depth=110, **kwargs)
        elif arch == 'resnet20':
            model = models.cifar.resnet(depth=20, **kwargs)
        elif arch == 'resnet32':
            model = models.cifar.resnet(depth=32, **kwargs)
        elif arch == 'resnet44':
            model = models.cifar.resnet(depth=44, **kwargs)
        elif arch == 'resnet56':
            model = models.cifar.resnet(depth=56, **kwargs)
        elif arch == 'resnet110':
            model = models.cifar.resnet(depth=110, **kwargs)
        elif arch == 'resnext-8x64d':
            model = models.cifar.resnext(depth=29, cardinality=8, widen_factor=4, **kwargs)
        elif arch == 'wrn-28-10-drop':
            model = models.cifar.wrn(depth=28, widen_factor=10, dropRate=0.3, **kwargs)
        else:
            raise NotImplementedError('Unknown arch {} for dataset {}'.format(arch, dataset))
    elif dataset == 'imagenet':
        model = eval('models.imagenet.{}(num_classes=1000, pretrained=\'imagenet\')'.format(arch))
    else:
        raise NotImplementedError('Unknown dataset {}'.format(dataset))

    return model
