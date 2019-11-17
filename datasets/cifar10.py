import torch.utils.data
import numpy as np
from torchvision import datasets, transforms


def make_loaders(train_batch=100, test_batch=100, num_val=5000):
    loaders = dict()
    # use pytorch default mnist data
    # save previous RNG state
    state = np.random.get_state()
    # fix random seed, thus we have the same train/val split every time
    np.random.seed(0)
    # restore previous RNG state for training && test
    perm = np.random.permutation(50000)
    np.random.set_state(state)
    assert perm[0] == 11841
    assert perm[1] == 19602
    # pytorch does not provide something like SubsetSampler
    # SubsetRandomSampler will shuffle validation set every time
    samplers = dict()
    # first num_val examples in training set is used as validation set
    samplers['train'] = torch.utils.data.sampler.SubsetRandomSampler(perm[num_val:])
    samplers['val'] = torch.utils.data.sampler.SubsetRandomSampler(perm[:num_val])
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (1., 1., 1.))
    for phase in ['train', 'trainval', 'val', 'test']:
        is_train = phase in ['train', 'trainval', 'val']
        eval_mode = phase in ['val', 'test']
        batch_size = test_batch if eval_mode else train_batch
        shuffle = phase == 'trainval'  # when phase=train, shuffle process happens in train_sampler
        sampler = None
        if phase in samplers:
            sampler = samplers[phase]

        if eval_mode:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize])

        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/pytorch-official/cifar10', train=is_train, download=True, transform=transform),
            batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=True, drop_last=True, num_workers=8)
        loaders[phase] = loader

    return loaders
