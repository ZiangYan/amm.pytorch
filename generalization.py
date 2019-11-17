#!/usr/bin/env python3
import sys
import os
import os.path as osp
import glog as log
import argparse
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import make_model
from models.utils import DropoutFreeze


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Use DeepFool distance proxy to improve generalization')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='base learning rate')
    parser.add_argument('--step-mult', nargs='+', default=[0.1, 0.01],
                        help='multiplier for step lr policy')
    parser.add_argument('--step-at', nargs='+', default=[50, 80],
                        help='step at specified epochs')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs to train')
    parser.add_argument('--lmbd', default=1, type=float,
                        help='lmbd in regularizer')
    parser.add_argument('--c', default=1e-3, type=float,
                        help='c in regularizer')
    parser.add_argument('--d', default=1e-1, type=float,
                        help='d in loss function')
    parser.add_argument('--num-group', default=16, type=int,
                        help='number of groups in group normalization')
    parser.add_argument('--scratch', action='store_true',
                        help='training from scratch or matconvnet pre-trained model')
    parser.add_argument('--no-hograd', action='store_true',
                        help='no high order gradient during training')
    parser.add_argument('--use-failed', action='store_true',
                        help='use perturbation for all examples including failed ones')
    parser.add_argument('--use-trainval', action='store_true',
                        help='use trainval set instead of train set for training')
    parser.add_argument('--num-train', default=0, type=int,
                        help='number of training samples used in training')
    parser.add_argument('--num-noisy', default=0, type=int,
                        help='number of training samples with random labels')
    parser.add_argument('--df-train-max-iter', default=5, type=int,
                        help='max iteration in deepfool in training')
    parser.add_argument('--df-test-max-iter', default=5, type=int,
                        help='max iteration in deepfool in test')
    parser.add_argument('--df-num-label', default=10, type=int,
                        help='number of target labels for DeepFool')
    parser.add_argument('--df-overshot', default=0.02, type=float,
                        help='overshot for DeepFool')
    parser.add_argument('--shrinkage', default='exp', type=str, choices=['lin', 'exp', 'invprop', 'margin'],
                        help='shrinkage function in regularizer')
    parser.add_argument('--aggregation', default='avg', type=str, choices=['avg', 'min'],
                        help='aggregation function in regularizer')
    parser.add_argument('--dropout-p', default=0.0, type=float,
                        help='dropout probability')
    parser.add_argument('--top-frac', default=0.2, type=float,
                        help='fraction used in aggregation function')
    parser.add_argument('--decay', default=0.0001, type=float,
                        help='weight decay')
    parser.add_argument('--train-batch', default=100, type=int,
                        help='train batch size')
    parser.add_argument('--batch', default=100, type=int,
                        help='actual train batch size (for gradient accumulation')
    parser.add_argument('--test-batch', default=100, type=int,
                        help='test batch size')
    parser.add_argument('--no-eval-r', action='store_true',
                        help='do not eval r in test model')
    parser.add_argument('--no-bp-prev-r', action='store_true',
                        help='do not BP into previous perturbation')
    parser.add_argument('--freeze-bn', action='store_true',
                        help='freeze bn weight & bias grad during regularizer calculation')
    parser.add_argument('--ce-freeze-bn', action='store_true',
                        help='freeze bn weight & bias grad during cross-entropy calculation')
    parser.add_argument('--tune-part', nargs='*', default=list(),
                        help='tuna part of network weights')
    parser.add_argument('--pos-only', action='store_true',
                        help='do not calculate regularizer for incorrectly classified samples')
    parser.add_argument('--exp-dir', default='output/debug', type=str,
                        help='directory to save models and logs')
    parser.add_argument('--save-every-epoch', action='store_true',
                        help='save model after each epoch')
    parser.add_argument('--weight', default='', type=str,
                        help='model weight file to load')
    parser.add_argument('--pretest', action='store_true',
                        help='evaluate model before training')
    parser.add_argument('--clip-val', default=50., type=float,
                        help='clip value of regularization term for numerical stability')
    parser.add_argument('--clip-grad', default=10, type=float,
                        help='clip gradient for numerical stability')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')
    parser.add_argument('--dataset', default='mnist', type=str,
                        choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet'],
                        help='which dataset to use')
    parser.add_argument('--arch', default='lenet', type=str,
                        help='network architecture, e.g., lenet')
    parser.add_argument('--save-gpu-mem', action='store_true',
                        help='empty cache in each iteration to save gpu memory, and this will lead to slower training')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.step_at = list(map(lambda e: int(e), args.step_at))
    args.step_mult = list(map(lambda e: float(e), args.step_mult))
    assert len(args.step_at) == len(args.step_mult)
    return args


class DeepFool(nn.Module):
    def __init__(self):
        super(DeepFool, self).__init__()

        self.num_label = args.df_num_label
        self.overshot = args.df_overshot

        # initialize net
        if args.dataset in ['mnist', 'cifar10', 'svhn']:
            num_classes = 10
        elif args.dataset in ['cifar100']:
            num_classes = 100
        elif args.dataset in ['imagenet']:
            num_classes = 1000
        else:
            raise NotImplementedError

        kwargs = {'num_classes': num_classes}
        if args.arch in ['LiuNet']:
            kwargs['num_group'] = args.num_group
        self.net = make_model(args.dataset, args.arch, **kwargs)

        # set dropout probability (if there are any dropout layers inside self.net)
        for m in self.net.modules():
            if isinstance(m, DropoutFreeze):
                m.p = args.dropout_p

        log.info(self.net)

        # misc
        self.eps = 5e-6 if args.dataset == 'mnist' else 1e-5  # protect norm against nan
        self.not_done = None

    def net_forward(self, image, no_grad=True, **kwargs):
        if no_grad:
            with torch.no_grad():
                logit = self.net(image, **kwargs)
        else:
            logit = self.net(image, **kwargs)
        return logit

    def inversenet_backward(self, image, target_label, no_grad=True, **kwargs):
        num_target_label = target_label.shape[1]
        batch_size = image.shape[0]
        image_shape = image.shape[1:]

        # expand image
        image = image.detach()
        image_expand = image.view(batch_size, 1, image_shape[0], image_shape[1], image_shape[2])
        image_expand = image_expand.repeat(1, num_target_label, 1, 1, 1)
        image_expand = image_expand.view(-1, image_shape[0], image_shape[1], image_shape[2])
        image_expand.requires_grad = True

        # expand additional inputs
        kwargs_expand = dict()
        for key, value, in kwargs.items():
            if value.ndimension() == 2:
                val_shape = value.shape[1]
                assert value.shape[0] == batch_size
                value_expand = value.view(batch_size, 1, val_shape)
                value_expand = value_expand.repeat(1, num_target_label, 1)
                value_expand = value_expand.view(-1, val_shape)
                kwargs_expand[key] = value_expand
            elif value.ndimension() == 4:
                val_shape = value.shape[1:]
                assert value.shape[0] == batch_size
                value_expand = value.view(batch_size, 1, *val_shape)
                value_expand = value_expand.repeat(1, num_target_label, 1, 1, 1)
                value_expand = value_expand.view(-1, *val_shape)
                kwargs_expand[key] = value_expand
            else:
                raise NotImplementedError

        logit_expand = self.net(image_expand, **kwargs_expand)
        logit_grad_expand = torch.zeros(target_label.numel(), logit_expand.shape[1]).to(image.device)
        logit_grad_expand[torch.arange(target_label.numel()), target_label.view(target_label.numel())] = 1.
        if not self.training or no_grad:
            grad_expand = torch.autograd.grad(logit_expand, image_expand,
                                              grad_outputs=logit_grad_expand, create_graph=False, only_inputs=True)[0]
        else:
            grad_expand = torch.autograd.grad(logit_expand, image_expand,
                                              grad_outputs=logit_grad_expand, create_graph=True, only_inputs=True)[0]

        grad_expand = grad_expand.view(batch_size, num_target_label, -1).transpose(1, 2)
        return grad_expand

    def project_boundary_polyhedron(self, grad, logit):
        batch_size = grad.size()[0]  # 100
        image_dim = grad.size()[1]  # 784 for mnist

        # project under l_2 norm
        # dist: |f| / |w|, then select target class with smallest dist
        # r: |f| / |w| ** 2 * w
        dist = torch.abs(logit) / torch.norm(grad + self.eps, p=2, dim=1).view(logit.size())
        selected_target = dist.argmin(dim=1)
        selected_dist = dist.gather(1, selected_target.view(batch_size, 1))
        selected_grad = grad.gather(
            2, selected_target.view(batch_size, 1, 1).expand(batch_size, image_dim, 1)).view(batch_size, image_dim)
        r = selected_dist * selected_grad / torch.norm(selected_grad + self.eps, p=2, dim=1).view(batch_size, 1)
        return r

    def clear_variable(self, name):
        try:
            var = self.__getattribute__(name)
            del var
        except AttributeError:
            pass

    def forward(self, image, target_label=None, no_grad=True, **kwargs):
        # set no_grad to True will not construct gradient graph of r, thus drastically reduce gpu memory usage
        # kwargs is used to maintain dropout mask

        # init some variables
        num_image = image.size()[0]
        image_shape = image.size()
        adv_image = image
        self.not_done = torch.ones(num_image, dtype=torch.uint8).to(image.device)
        max_iter = args.df_train_max_iter if self.training else args.df_test_max_iter
        r = 0

        # get label and target_label
        logit = self.net_forward(adv_image, no_grad=no_grad, **kwargs)
        orig_pred = logit.argmax(dim=1)
        pred = orig_pred
        if target_label is None:
            # untargeted attack
            attack_type = 'untargeted'
            target_label = torch.sort(-logit, dim=1)[1]
            target_label = target_label.data[:, :self.num_label]
        else:
            # targeted attack
            attack_type = 'targeted'
            target_label = torch.cat([orig_pred.view(num_image, 1), target_label.view(num_image, 1)], dim=1)

        for iteration in range(max_iter):
            # get logit diff (a.k.a, f - f0)
            logit_diff = logit - logit.gather(1, pred.view(num_image, 1))

            # get grad diff (a.k.a, w - w0)
            if attack_type == 'targeted':
                target_label = torch.cat([pred.view(num_image, 1), target_label[:, 1].view(num_image, 1)], dim=1)
            grad = self.inversenet_backward(adv_image, target_label.contiguous(), no_grad=no_grad, **kwargs)
            grad_diff = grad - grad[:, :, 0].contiguous().view(grad.size()[0], grad.size()[1], 1).expand_as(grad)

            r_this_step = \
                self.project_boundary_polyhedron(grad_diff[:, :, 1:], logit_diff.gather(1, target_label[:, 1:]))

            # if an image is already successfully fooled, no more perturbation should be applied to it
            r_this_step = r_this_step * self.not_done.float().view(num_image, 1)

            # add some overshot
            r_this_step = (1 + args.df_overshot) * r_this_step.view(image_shape)

            # accumulate r
            r = r + r_this_step

            # add r_this_step to adv_image
            adv_image = adv_image + r_this_step

            # stop gradient for efficient training (if necessary)
            if not self.training or args.no_bp_prev_r:
                adv_image = adv_image.detach()
                adv_image.requires_grad = True

            # test whether we have successfully fooled these images
            logit = self.net_forward(adv_image, no_grad=no_grad, **kwargs)
            pred = logit.argmax(dim=1)
            if attack_type == 'untargeted':
                self.not_done = self.not_done * pred.eq(orig_pred)
            else:
                self.not_done = self.not_done * ~(pred.eq(target_label[:, 1]))

            if torch.all(~self.not_done).item():
                # break if already fooled all images
                break

        return r


def test(model, phases='test'):
    model.eval()
    result = dict()
    if isinstance(phases, str):
        phases = [phases]
    for phase in phases:
        log.info('Evaluating model, phase = {}'.format(phase))
        loader = loaders[phase]

        num_image = len(loader) * loader.batch_size
        log.info('Found {} images in phase {}'.format(num_image, phase))

        acc_all = torch.zeros(num_image)
        ce_loss_all = torch.zeros(num_image)
        r_norm_all = torch.zeros(num_image)

        for index, (image, label) in enumerate(loader):
            # get one batch
            image = image.to(device)
            label = label.long().to(device)
            selected = torch.arange(index * args.test_batch, index * args.test_batch + image.shape[0])

            # calculate cross entropy
            with torch.no_grad():
                logit = model.net(image)
            ce_loss = F.cross_entropy(logit, label)
            pred = logit.argmax(dim=1)

            # calculate accuracy
            acc = pred == label

            # calculate perturbation norm
            if args.no_eval_r:
                r_norm = torch.zeros(image.shape[0])
            else:
                r = model(image, no_grad=True)
                r_norm = r.view(r.shape[0], -1).norm(dim=1)

            # save results to arrays
            for key in ['acc', 'ce_loss', 'r_norm']:
                value_all = eval('{}_all'.format(key))
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()

            n = (index + 1) * args.test_batch
            if n % 2000 == 0:
                log.info('Evaluating {} set {} / {}'.format(phase, n, num_image))
                log.info('   r_norm: {:.4f}'.format((r_norm_all.sum() / n).item()))
                log.info('  ce_loss: {:.4f}'.format((ce_loss_all.sum() / n).item()))
                log.info('      acc: {:.4f}'.format((acc_all.sum() / n).item()))

        # cut last batch
        remainder = args.test_batch - image.shape[0]
        if remainder > 0:
            r_norm_all = r_norm_all[:-remainder]
            ce_loss_all = ce_loss_all[:-remainder]
            acc_all = acc_all[:-remainder]
            log.info('Cut number of {} images from {} to {}, since last batch contains only {} images'.format(
               phase, num_image, num_image-remainder, image.shape[0]))

        log.info('Performance on {} set ({} images) is:'.format(phase, r_norm_all.shape[0]))
        log.info('   r_norm: {:.4f}'.format(r_norm_all.mean().item()))
        log.info('  ce_loss: {:.4f}'.format(ce_loss_all.mean().item()))
        log.info('      acc: {:.4f}'.format(acc_all.mean().item()))

        result['{}_acc'.format(phase)] = acc_all.mean().item()
        result['{}_r_norm'.format(phase)] = r_norm_all.mean().item()
    log.info('Performance of current model is:')
    for phase in ['train', 'val', 'test']:
        if '{}_acc'.format(phase) in result:
            log.info('  {} r_norm: {:.4f}'.format(phase, result['{}_r_norm'.format(phase)]))
            log.info('     {} acc: {:.4f}'.format(phase, result['{}_acc'.format(phase)]))
    return result


def train(model):
    num_epoch = args.epochs
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=0.9)
    # exclude some weights in regularizer BP process
    keep_grads = dict()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                keep_grads[id(m.weight)] = 1
                keep_grads[id(m.bias)] = 1
    if len(args.tune_part) > 0:
        for name, value in model.named_parameters():
            exclude = True
            for t in args.tune_part:
                if t in name:
                    exclude = False
                    break
            if exclude:
                keep_grads[id(value)] = 1
    log.info('Exclude {} weights from regularizer backward ({} parameters in total)'.format(
        len(keep_grads), len(list(model.parameters()))))

    # select loader
    if args.use_trainval:
        loader = loaders['trainval']
    else:
        loader = loaders['train']
    num_image = len(loader) * args.batch
    log.info('Found {} images'.format(num_image))

    best_acc = 0
    for epoch_idx in range(num_epoch):
        log.info('Training for {} epoch'.format(epoch_idx))

        # adjust learning rate
        if epoch_idx in args.step_at:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                new_lr = args.lr * args.step_mult[args.step_at.index(epoch_idx)]
                param_group['lr'] = new_lr
                log.info('Epoch {}, cut learning rate from {:g} to {:g}'.format(epoch_idx, lr, new_lr))

        r_norm_all = torch.zeros(num_image)
        ce_loss_all = torch.zeros(num_image)
        reg_all = torch.zeros(num_image)
        acc_all = torch.zeros(num_image)
        success_all = torch.zeros(num_image)
        grad_all = torch.zeros(num_image)
        clip_grad_all = torch.zeros(num_image)

        # gradient accumulation
        assert args.train_batch >= args.batch and args.train_batch % args.batch == 0

        for index, (image, label) in enumerate(loader):
            model.train()

            # unfreeze all non-deterministic layers for cross entropy training
            for m in model.modules():
                if isinstance(m, DropoutFreeze):
                    m.freeze = False
                if isinstance(m, nn.BatchNorm2d):
                    if not args.ce_freeze_bn:
                        m.train()
                    else:
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

            image = image.to(device)
            label = label.long().to(device)
            selected = torch.arange(index * args.batch, index * args.batch + image.shape[0])

            # accuracy
            logit = model.net(image)
            pred = logit.argmax(dim=1)
            acc_all[selected] = (pred == label).detach().float().cpu()

            # cross entropy loss
            ce_loss = F.cross_entropy(logit, label) * args.batch / args.train_batch
            ce_loss.backward()
            ce_loss_all[selected] = ce_loss.detach().cpu()

            if args.lmbd > 0:
                # enforce a deterministic behavior of the network && freeze grads
                kwargs = dict()
                for m in model.modules():
                    if isinstance(m, DropoutFreeze):
                        m.freeze = True
                        kwargs[m.name + '_mask'] = m.mask.to(image.device)
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

                if len(keep_grads) > 0:
                    for name, value in model.named_parameters():
                        if id(value) in keep_grads and value.grad is not None:
                            keep_grads[id(value)] = value.grad.clone()

                # re-calculate pred in the deterministic manner, and split pos/neg
                pred = model.net_forward(image, no_grad=True, **kwargs).argmax(dim=1)
                pos_index = torch.nonzero(pred == label)[:, 0].cpu()
                neg_index = torch.nonzero(pred != label)[:, 0].cpu()
                pos_selected = index * args.batch + pos_index
                neg_selected = index * args.batch + neg_index

                if pos_index.numel() > 0:
                    pos_kwargs = dict()
                    for key, value in kwargs.items():
                        pos_kwargs[key] = value[pos_index]
                    pos_image = image[pos_index]
                    pos_label = label[pos_index]
                    if args.aggregation == 'min':
                        # we may re-calculate r with grad_fn on selected images later
                        r = model(image=pos_image, target_label=None, no_grad=True, **pos_kwargs)
                    elif args.aggregation == 'avg':
                        r = model(image=pos_image, target_label=None, no_grad=False, **pos_kwargs)
                    else:
                        raise NotImplementedError('Unknown aggregation function {}'.format(args.aggregation))

                    r_norm = r.view(r.shape[0], -1).norm(dim=1)
                    r_norm_all[pos_selected] = r_norm.detach().cpu()

                    # select some images to perform BP
                    success = (~model.not_done).nonzero().flatten().to(image.device)
                    success_all[pos_selected] = (~model.not_done).detach().float().cpu()
                    if args.use_failed or success.numel() > 0:
                        if not args.use_failed:
                            # exclude failed images
                            pos_image = pos_image.index_select(0, success)
                            pos_label = pos_label.index_select(0, success)
                            r_norm = r_norm.index_select(0, success)
                            for key, value in pos_kwargs.items():
                                pos_kwargs[key] = value.index_select(0, success)

                        # re-calculate r to add grad_fn to selected images
                        sort_idx = r_norm.sort()[1]  # sort_idx[0] corresponds to the smallest r_norm
                        if args.aggregation == 'min':
                            # find worst case
                            all_labels = pos_label.unique()
                            perclass_index = list()
                            for current_label in all_labels:
                                index_current_label = (pos_label == current_label).nonzero().flatten()
                                t = r_norm.index_select(0, index_current_label).argmin()
                                perclass_index.append(index_current_label[t].item())
                            # intersect with top args.top_frac% smallest r norm set
                            num_top = max(1, int(image.shape[0] * args.top_frac))
                            min_index = np.intersect1d(perclass_index, sort_idx[:num_top].detach().cpu().numpy())
                            if min_index.size == 0:
                                # this may happen if many samples are clipped by args.clip_val
                                min_index = [perclass_index[0]]
                            min_index = torch.LongTensor(min_index).to(image.device)

                            # do re-calculation
                            pos_image = pos_image.index_select(0, min_index)
                            for key, value in pos_kwargs.items():
                                pos_kwargs[key] = value.index_select(0, min_index)
                            r = model(image=pos_image, target_label=None, no_grad=False, **pos_kwargs)
                            r_norm = r.view(r.shape[0], -1).norm(dim=1)
                        else:
                            # no re-calculation for other aggregation methods
                            pass

                        if args.shrinkage == 'invprop':
                            # numerical protection for invprop function
                            pos_reg = args.lmbd * (1. / (1. + args.c * r_norm))
                            pos_reg = torch.clamp(pos_reg, max=args.clip_val)
                        elif args.shrinkage == 'exp':
                            pos_reg = args.lmbd * torch.exp(-args.c * r_norm)
                            pos_reg = torch.clamp(pos_reg, max=args.clip_val)
                        elif args.shrinkage == 'lin':
                            pos_reg = args.lmbd * (-args.c * r_norm)
                        elif args.shrinkage == 'margin':
                            pos_reg = args.lmbd * torch.clamp(args.c - r_norm, min=0)
                        else:
                            raise NotImplementedError

                        pos_reg = pos_reg.sum() / args.train_batch
                        reg_all[pos_selected] = pos_reg.detach().cpu()

                        # BP
                        pos_reg.backward()

                if neg_index.numel() > 0 and not args.pos_only:
                    neg_kwargs = dict()
                    for key, value in kwargs.items():
                        neg_kwargs[key] = value[neg_index]
                    neg_image = image[neg_index]
                    neg_label = label[neg_index]
                    if args.aggregation == 'min':
                        # we may re-calculate r with grad_fn on selected images later
                        r = model(image=neg_image, target_label=neg_label, no_grad=True, **neg_kwargs)
                    elif args.aggregation == 'avg':
                        r = model(image=neg_image, target_label=neg_label, no_grad=False, **neg_kwargs)
                    else:
                        raise NotImplementedError('Unknown aggregation function {}'.format(args.aggregation))

                    r_norm = r.view(r.shape[0], -1).norm(dim=1)
                    r_norm_all[neg_selected] = r_norm.detach().cpu()

                    # select some images to perform BP
                    success = (~model.not_done).nonzero().flatten().to(image.device)
                    success_all[neg_selected] = (~model.not_done).detach().float().cpu()
                    if args.use_failed or success.numel() > 0:
                        if not args.use_failed:
                            # exclude failed images
                            neg_image = neg_image.index_select(0, success)
                            neg_label = neg_label.index_select(0, success)
                            r_norm = r_norm.index_select(0, success)
                            for key, value in neg_kwargs.items():
                                neg_kwargs[key] = value.index_select(0, success)

                        # re-calculate r to add grad_fn to selected images
                        sort_idx = r_norm.sort()[1]  # sort_idx[0] is the smallest r_norm
                        if args.aggregation == 'min':
                            # find worst case
                            all_labels = neg_label.unique()
                            perclass_index = list()
                            for current_label in all_labels:
                                index_current_label = (neg_label == current_label).nonzero().flatten()
                                t = r_norm.index_select(0, index_current_label).argmax()
                                perclass_index.append(index_current_label[t].item())
                            # intersect with top args.top_frac% largest r norm set
                            num_top = max(1, int(image.shape[0] * args.top_frac))
                            min_index = np.intersect1d(perclass_index, sort_idx[-num_top:].detach().cpu().numpy())
                            if min_index.size == 0:
                                # this may happen if many samples are clipped by args.clip_val
                                min_index = [perclass_index[0]]
                            min_index = torch.LongTensor(min_index).to(image.device)

                            # do re-calculation
                            neg_image = neg_image.index_select(0, min_index)
                            neg_label = neg_label.index_select(0, min_index)
                            for key, value in neg_kwargs.items():
                                neg_kwargs[key] = value.index_select(0, min_index)
                            r = model(image=neg_image, target_label=neg_label, no_grad=False, **neg_kwargs)
                            r_norm = r.view(r.shape[0], -1).norm(dim=1)
                        else:
                            # no re-calculation for other aggregation methods
                            pass

                        if args.shrinkage == 'invprop':
                            # numerical protection for invprop function
                            neg_reg = torch.clamp(r_norm, max=(1. - args.lmbd / args.clip_val) / args.d)
                            neg_reg = args.lmbd * (1. / (1. - args.d * neg_reg))
                        elif args.shrinkage == 'exp':
                            # numerical protection for exp function, add 1 to prevent args.clip_val=0
                            neg_reg = torch.clamp(r_norm, max=np.log(args.clip_val / args.lmbd + 1) / args.d)
                            neg_reg = args.lmbd * torch.exp(args.d * neg_reg)
                            neg_reg = torch.clamp(neg_reg, max=args.clip_val)
                        elif args.shrinkage == 'lin':
                            neg_reg = args.lmbd * args.d * r_norm
                        elif args.shrinkage == 'margin':
                            neg_reg = args.lmbd * torch.clamp(args.c + r_norm, min=0)
                        else:
                            raise NotImplementedError

                        neg_reg = neg_reg.sum() / args.train_batch
                        reg_all[neg_selected] = neg_reg.detach().cpu()

                        # BP
                        neg_reg.backward()

                # restore grads
                if len(keep_grads) > 0:
                    for name, value in model.named_parameters():
                        if id(value) in keep_grads and value.grad is not None:
                            value.grad[:] = keep_grads[id(value)]

            # clip grad norm
            grad_all[selected] = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, norm_type=2)
            for p in model.parameters():
                if p.grad is not None:
                    clip_grad_all[selected] += p.grad.detach().norm(2) ** 2
            clip_grad_all[selected] = torch.sqrt(clip_grad_all[selected])

            # number of example visited in this epoch
            n = (index + 1) * args.batch
            if (index + 1) % (args.train_batch / args.batch) == 0:
                log.info('Processing {} / {}'.format(n, num_image))
                keys = ['r_norm', 'ce_loss', 'reg', 'acc', 'grad', 'clip_grad', 'success']
                max_len = max(list(map(lambda t: len(t), keys)))
                for key in keys:
                    value = eval('{}_all'.format(key))
                    epoch_mean = (value.sum() / n).item()
                    selected = torch.arange(n - args.train_batch, n)
                    batch_mean = value[selected].mean().item()
                    log.info(' ' * (max_len + 1 - len(key)) + key +
                             ': epoch {:.4f}, batch {:.4f}'.format(epoch_mean, batch_mean))

                # update weights
                optimizer.step()
                optimizer.zero_grad()

            if args.save_gpu_mem:
                torch.cuda.empty_cache()

        # evaluate and save current model
        log.info('Evaluating epoch {}'.format(epoch_idx))
        if args.use_trainval:
            if args.dataset in ['mnist', 'cifar10', 'cifar100', 'svhn']:
                result = test(model, phases='test')
            elif args.dataset == 'imagenet':
                result = test(model, phases='val')
            else:
                raise NotImplementedError('Unknown dataset {}'.format(args.dataset))
        else:
            if args.dataset in ['mnist', 'cifar10', 'cifar100', 'svhn']:
                result = test(model, phases=['val', 'test'])
            elif args.dataset == 'imagenet':
                result = test(model, phases='val')
            else:
                raise NotImplementedError('Unknown dataset {}'.format(args.dataset))

        # save model
        if args.save_every_epoch or (epoch_idx + 1 in args.step_at):
            model_fname = osp.join(args.exp_dir, 'epoch_{}.model'.format(epoch_idx))
            if not osp.exists(osp.dirname(model_fname)):
                os.makedirs(osp.dirname(model_fname))
            torch.save(model.state_dict(), model_fname)
            log.info('Model of epoch {} saved to {}'.format(epoch_idx, model_fname))

        if args.dataset in ['mnist', 'cifar10', 'cifar100', 'svhn']:
            test_phase = 'test' if args.use_trainval else 'val'
        elif args.dataset == 'imagenet':
            test_phase = 'val'
        else:
            raise NotImplementedError('Unknown dataset {}'.format(args.dataset))
        if result['{}_acc'.format(test_phase)] > best_acc:
            best_acc = result['{}_acc'.format(test_phase)]
            model_fname = osp.join(args.exp_dir, 'best.model'.format(epoch_idx))
            if not osp.exists(osp.dirname(model_fname)):
                os.makedirs(osp.dirname(model_fname))
            torch.save(model.state_dict(), model_fname)
            log.info('Best model (epoch {}) saved to {}'.format(epoch_idx, model_fname))


def main():
    model = DeepFool()

    if not args.scratch:
        if args.dataset != 'imagenet':
            assert osp.exists(args.weight)
            state_dict = torch.load(args.weight, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            log.info('Training from pre-trained weights {}'.format(args.weight))
        else:
            log.info('Training from pre-trained imagenet model')
    else:
        log.info('Training from scratch')

    model.to(device)

    if args.pretest:
        log.info('Evaluating performance before fine-tune')
        if args.use_trainval:
            if args.dataset in ['mnist', 'cifar10', 'cifar100', 'svhn']:
                test(model, phases='test')
            elif args.dataset == 'imagenet':
                test(model, phases='val')
            else:
                raise NotImplementedError('Unknown dataset {}'.format(args.dataset))
        else:
            if args.dataset in ['mnist', 'cifar10', 'cifar100', 'svhn']:
                test(model, phases=['val', 'test'])
            elif args.dataset == 'imagenet':
                test(model, phases='val')
            else:
                raise NotImplementedError('Unknown dataset {}'.format(args.dataset))


    log.info('Training network')
    train(model)

    log.info('Saving model')
    fname = osp.join(args.exp_dir, 'final.model')
    if not osp.exists(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname))
    torch.save(model.state_dict(), fname)
    log.info('Final model saved to {}'.format(fname))


def set_log_file(fname):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess

    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def print_args():
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname


if __name__ == '__main__':
    # before going to the main function, we do following things:
    # 1. setup output directory
    # 2. make global variables: args, model (on cpu), loaders and device

    # 1. setup output directory
    args = parse_args()

    log.info('Called with args:')
    log.info(args)

    args.exp_dir = osp.join(args.exp_dir, get_random_dir_name())
    if not osp.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # set log file
    set_log_file(osp.join(args.exp_dir, 'run.log'))

    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args()

    # dump config.json
    with open(osp.join(args.exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # backup scripts
    fname = __file__
    if fname.endswith('pyc'):
        fname = fname[:-1]
    os.system('cp {} {}'.format(fname, args.exp_dir))
    os.system('cp -r datasets models {}'.format(args.exp_dir))

    # 2. make global variables
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:0')

    if args.dataset == 'mnist':
        from datasets.mnist import make_loaders
        loaders = make_loaders(train_batch=args.batch, test_batch=args.test_batch,
                               num_train=args.num_train, num_noisy=args.num_noisy)
    elif args.dataset == 'cifar10':
        from datasets.cifar10 import make_loaders
        loaders = make_loaders(train_batch=args.batch, test_batch=args.test_batch)
    elif args.dataset == 'cifar100':
        from datasets.cifar100 import make_loaders
        loaders = make_loaders(train_batch=args.batch, test_batch=args.test_batch)
    elif args.dataset == 'svhn':
        from datasets.svhn import make_loaders
        loaders = make_loaders(train_batch=args.batch, test_batch=args.test_batch)
    elif args.dataset == 'imagenet':
        from datasets.imagenet import make_loaders
        loaders = make_loaders(train_batch=args.batch, test_batch=args.test_batch)
    else:
        raise NotImplementedError('Unknown dataset {}'.format(args.dataset))

    # do the business
    main()
