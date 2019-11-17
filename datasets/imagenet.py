import torch.utils.data
import numpy as np
from torchvision import datasets, transforms


def make_loaders(train_batch=100, test_batch=100):
    loaders = dict()
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    loaders['train'] = torch.utils.data.DataLoader(
        datasets.ImageFolder('data/pytorch-official/imagenet/train', transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=train_batch, shuffle=True, num_workers=8, pin_memory=True)

    loaders['val'] = torch.utils.data.DataLoader(
        datasets.ImageFolder('data/pytorch-official/imagenet/val', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=test_batch, shuffle=False, num_workers=8, pin_memory=True)

    return loaders
