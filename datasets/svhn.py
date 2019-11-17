import torch.utils.data
import numpy as np
from torchvision import datasets, transforms


def make_loaders(train_batch=100, test_batch=100, num_val=6000):
    loaders = dict()
    # use pytorch default mnist data
    # save previous RNG state
    state = np.random.get_state()
    # fix random seed, thus we have the same train/val split every time
    np.random.seed(0)
    # restore previous RNG state for training && test
    perm = np.random.permutation(604388)
    assert perm[0] == 304716
    assert perm[1] == 163666
    np.random.set_state(state)
    # pytorch does not provide something like SubsetSampler
    # SubsetRandomSampler will shuffle validation set every time
    samplers = dict()
    # first num_val examples in training set is used as validation set
    samplers['train'] = torch.utils.data.sampler.SubsetRandomSampler(perm[num_val:])
    samplers['val'] = torch.utils.data.sampler.SubsetRandomSampler(perm[:num_val])
    # normalize = transforms.Normalize((0.4376 , 0.4437 , 0.4728), (0.1980, 0.2010, 0.1970))
    normalize = transforms.Normalize((0.4376, 0.4437, 0.4728), (1., 1., 1.))

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          # transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    data_dirname = './data/pytorch-official/svhn'
    d1 = datasets.SVHN(data_dirname, split='train', download=True, transform=train_transform)
    d2 = datasets.SVHN(data_dirname, split='extra', download=True, transform=train_transform)
    trainval_dataset = torch.utils.data.ConcatDataset([d1, d2])
    test_dataset = datasets.SVHN(data_dirname, split='test', download=True, transform=test_transform)
    for phase in ['train', 'trainval', 'val', 'test']:
        eval_mode = phase in ['val', 'test']
        batch_size = test_batch if eval_mode else train_batch
        shuffle = phase == 'trainval'  # when phase=train, shuffle process happens in train_sampler
        sampler = None
        if phase in samplers:
            sampler = samplers[phase]

        dataset = trainval_dataset if phase != 'test' else test_dataset
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                             sampler=sampler, pin_memory=True, drop_last=False, num_workers=8)
        loaders[phase] = loader

    return loaders
