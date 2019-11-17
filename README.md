# amm.pytorch
Code for our T-PAMI 2019 paper [Adversarial Margin Maximization Networks](https://ieeexplore.ieee.org/document/8877866).

Our paper is also available on [arxiv](https://arxiv.org/abs/1911.05916).

## Environments
* Python 3.5
* PyTorch 1.1.0
* torchvision 0.2.2
* glog 0.3.1

## Datasets and Reference Models
We use MNIST, CIFAR-10/100, SVHN and ImageNet in pytorch's default format. One can check scripts in ```datasets/``` for more details.

## Usage
To train a MLP800 on MNIST using cross entropy loss:

```
python3 generalization.py --scratch --dataset mnist --arch mlp800 --use-trainval --lmbd 0 --lr 0.005 
```

To train a MLP800 on MNIST using AMM:

```
python3 generalization.py --d 1.0 --scratch --shrinkage exp --dataset mnist --c 2.0 --arch mlp800 --use-trainval --lmbd 32.0 --aggregation min --lr 0.005 
```


## Citation
Please cite our work in your publications if it helps your research:

```
@article{yan2019adversarial,
  title={Adversarial Margin Maximization Networks},
  author={Yan, Ziang and Guo, Yiwen and Zhang, Changshui},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2019},
  publisher={IEEE}
}
```
