# Resolution Adaptive Networks for Efficient Inference (CVPR2020)
[Le Yang*](https://github.com/yangle15), [Yizeng Han*](https://github.com/thuallen), [Xi Chen*](https://github.com/FateDawnLeon), Shiji Song, [Jifeng Dai](https://github.com/daijifeng001), [Gao Huang](https://github.com/gaohuang)

This repository contains the implementation of the paper, '[Resolution Adaptive Networks for Efficient Inference](https://arxiv.org/pdf/2003.07326.pdf)'. The proposed Resolution Adaptive Networks (RANet) conduct the adaptive inferece by exploiting the ``spatial redundancy`` of input images. Our motivation is that low-resolution representations are sufficient for classifying easy samples containing large objects with prototypical features, while only some hard samples need spatially detailed information, which can be demonstrated by the follow figure.

<div align=center><img width="380" height="410" src="https://github.com/yangle15/RANet-pytorch/blob/master/imgs/RANet_overview.png"/></div>

## Results

<div align=center><img width="800" height="230" src="https://github.com/yangle15/RANet-pytorch/blob/master/imgs/anytime_results.png"/></div>

Accuracy (top-1) of anytime prediction models as a function of computational budget on the CIFAR-10 (left), CIFAR-100
(middle) and ImageNet (right) datasets. Higher is better.

<div align=center><img width="800" height="230" src="https://github.com/yangle15/RANet-pytorch/blob/master/imgs/dynamic_results.png"/></div>

Accuracy (top-1) of budgeted batch classification models as a function of average computational budget per image the on CIFAR-
10 (left), CIFAR-100 (middle) and ImageNet (right) datasets. Higher is better.

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
python main.py --arch RANet --gpu '0' --data-root YOUR_DATA_PATH --data 'cifar10' --step 2 --nChannels 16 --stepmode 'lg' --scale-list '1-2-3' --grFactor '4-2-1' --bnFactor '4-2-1'
```

### Train a RANet on ImageNet
* Modify the train_imagenet.sh to config your path to the dataset, your GPU devices and your saving directory. Then run
```sh
bash train_imagenet.sh
```

* You can train your RANet with other configurations.
```sh
python main.py --arch RANet --gpu '0,1,2,3' --data-root YOUR_DATA_PATH --data 'ImageNet' --step 8 --growthRate 16 --nChannels 32 --stepmode 'even' --scale-list '1-2-3-4' --grFactor '4-2-2-1' --bnFactor '4-2-2-1'
```



### Citation
If you find this work useful or use our codes in your own research, please use the following bibtex:
```
@inproceedings{yang2020resolution,
  title={Resolution Adaptive Networks for Efficient Inference},
  author={Yang, Le and Han, Yizeng and Chen, Xi and Song, Shiji and Dai, Jifeng and Huang, Gao},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

### Contact
If you have any questions, please feel free to contact the authors. 

Le Yang: yangle15@mails.tsinghua.edu.cn

Yizeng Han: [hanyz18@mails.tsinghua.edu.cn](mailto:hanyz18@mails.tsinghua.edu.cn)

### Acknowledgments
We use the pytorch implementation of MSDNet in our experiments. The code can be found [here](https://github.com/kalviny/MSDNet-PyTorch).



