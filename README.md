# Resolution Adaptive Networks for Efficient Inference (CVPR2020)

This repository contains the implementation of our CVPR 2020 paper, 'Resolution Adaptive Networks for Efficient Inference'. The proposed Resolution Adaptive Networks (RANet) conduct the adaptive inferece by exploiting the ``spatial redundancy`` of input images. Our motivation is that low-resolution representations are sufficient for classifying easy samples containing large objects with prototypical features, while only some hard samples need spatially detailed information, which can be demonstrated by the follow figure.

<div align=center><img width="410" height="350" src="https://github.com/yangle15/RANet-pytorch/blob/master/imgs/RANet_overview.png"/></div>

## Results

<div align=center><img width="800" height="300" src="https://github.com/yangle15/RANet-pytorch/blob/master/imgs/anytime_results.png"/></div>
<div align=center><img width="800" height="300" src="https://github.com/yangle15/RANet-pytorch/blob/master/imgs/dynamic_results.png"/></div>

## Dependencies:

* Python3

* PyTorch >= 1.0

## Usage
We Provide shell scripts for training a RANet on CIFAR and ImageNet.

### Train a RANet on CIFAR
* Modify the train_cifar.sh to config your path to the dataset, your GPU devices and your saving directory. Then run
```sh
bash train_cifar.sh
```

* You can train your RANet with other configurations.
```sh
python main.py --arch RANet --data-root {your data root} --data 'cifar10' --step 2 --nChannels 16 --stepmode 'lg' --scale-list '1-2-3' --grFactor '4-2-1' --bnFactor '4-2-1'
```
 
### Train a RANet on ImageNet
Modify the run_GE.sh to config your path to the dataset, your GPU devices and your saving directory. Then run
```sh
bash train_imagenet.sh
```

Or, you can train your RANet with other configurations.
```sh
python main.py --arch RANet --data-root {your data root} --data 'ImageNet' --step 8 --growthRate 16 --nChannels 32 --stepmode 'even' --scale-list '1-2-3-4' --grFactor '4-2-2-1' --bnFactor '4-2-2-1'
```



### Citation
If you find this work useful or use our codes in your own research, please use the following bibtex:
```sh
git clone https://github.com/github_username/repo.git
```

