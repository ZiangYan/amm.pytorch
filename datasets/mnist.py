import torch.utils.data
import numpy as np
from torchvision import datasets, transforms


class NoisyMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, num_noisy=0):
        super(NoisyMNIST, self).__init__(root, train, transform, target_transform, download)

        # save previous RNG state
        state = np.random.get_state()
        # fix random seed, thus we have the same noise at each time
        np.random.seed(1)
        # restore previous RNG state for training && test
        perm = np.random.permutation(60000)
        assert perm[0] == 15281
        assert perm[1] == 21435

        # add noise to label
        if num_noisy > 0 and self.train:
            labels = np.random.randint(0, 10, num_noisy, dtype=np.int64)
            self.train_labels[perm[:num_noisy]] = torch.from_numpy(labels).long()
        np.random.set_state(state)


def make_loaders(train_batch=100, test_batch=100, num_train=60000, num_val=5000, num_noisy=0):
    loaders = dict()
    # use pytorch default mnist data
    # save previous RNG state
    state = np.random.get_state()
    # fix random seed, thus we have the same train/val split every time
    np.random.seed(0)
    # restore previous RNG state for training && test
    perm = np.random.permutation(60000)
    np.random.set_state(state)
    assert perm[0] == 3048
    assert perm[1] == 19563
    # pytorch does not provide something like SubsetSampler
    # SubsetRandomSampler will shuffle validation set every time
    samplers = dict()
    if num_train == 0:
        # use full training set
        samplers['train'] = torch.utils.data.sampler.SubsetRandomSampler(perm[num_val:])
    else:
        # use part of training set
        samplers['train'] = torch.utils.data.sampler.SubsetRandomSampler(perm[num_val:num_val+num_train])
        samplers['trainval'] = torch.utils.data.sampler.SubsetRandomSampler(perm[:num_train])

    samplers['val'] = torch.utils.data.sampler.SubsetRandomSampler(perm[:num_val])
    normalize = transforms.Normalize((0.1307,), (1.,))
    for phase in ['train', 'trainval', 'val', 'test']:
        is_train_data = phase in ['train', 'trainval', 'val']
        eval_mode = phase in ['val', 'test']
        batch_size = test_batch if eval_mode else train_batch
        shuffle = phase == 'trainval' if num_train is None else False  # shuffle happens in subset sampler
        sampler = None
        if phase in samplers:
            sampler = samplers[phase]

        loader = torch.utils.data.DataLoader(
            NoisyMNIST('data/pytorch-official/mnist', train=is_train_data, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), normalize]), num_noisy=num_noisy),
            batch_size=batch_size, shuffle=shuffle, sampler=sampler, pin_memory=True, drop_last=False, num_workers=8)
        loaders[phase] = loader

    return loaders
