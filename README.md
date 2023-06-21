# Multi-way Backpropagation for Training Compact Deep Neural Networks #

Training code for Multi-way BP. Both the Pytorch and Torch implementations are available.

## PyTorch Implementation ##

### Requirements ###

- Pytorch=1.0.0
- python=2.7

### Train Method ###  

1. Prepare data  

	Download the training data (e.g., CIFAR-10) and put them to your own directory.  

2. Train deep models with Multi-way BP
```
 cd multiwaybp-pytorch
 python main.py
```

You may refer to options.py for more argument.

Some Key arguments:
- dataPath : path for loading data set
- save_path : path for saving model file
- nGPU : number of GPUs to use (support multi-gpu) (Default: 1)
- netType : choose the network type as baseline model
- pivotSet : where to add aux-classifier

## Torch Implementation ##

### Requirements ###

See the [installation instructions](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md "installation") for a step-by-step guide.

- Install [Torch](http://torch.ch/ "torch") on a machine with CUDA GPU
- Install [cuDNN](https://developer.nvidia.com/cudnn "cudnn") and the corresponding bindings in Torch

If you already have Torch installed, update the luarocks ```nn```, ```cunn``` and ```cudnn```.

### Training Method ###  

1. Prepare data  

	Download the training data (e.g., CIFAR-10) and put them to your own directory.  

2. Train deep models with Multi-way BP
```
 cd multiwaybp-torch
 th train.lua
```

### Testing Method ###

1. Test pre-trained models

- [CIFAR10-MwResNet-56-2](https://disk.yandex.com/d/zMvzifB0vcyGA "MwResNet-56-2")
- [CIFAR10-MwResNet-56-5](https://disk.yandex.com/d/k1_34p-qvjdCT "MwResNet-56-5")
- [CIFAR10-MwResNet-26-2/10](https://disk.yandex.com/d/g-fKiJdKvcyJH "MwResNet-26-2/10")
- [CIFAR100-MwResNet-56-2](https://disk.yandex.com/d/9GTk0HrYvcyK6 "MwResNet-56-2")
- [CIFAR100-MwResNet-56-5](https://disk.yandex.com/d/NqIb0RYyvcyKo "MwResNet-56-5")
- [CIFAR100-MwResNet-26-2/10](https://disk.yandex.com/d/W8S5Cp3hvcyLT "MwResNet-26-2/10")

To test the performance of the MwResNet models, please download the pre-trained models and move them into the directory ``` ./pretrained ```.
Then you can run the script [test.lua](https://github.com/tanmingkui/multiwaybp/tree/master/multiwaybp-torch/test.lua "testing"). For example:

```
th test.lua -dataset cifar10 -model cifar10-mwresnet-26-2-wide-10 
```

2. Test intermediate models

During the training, **Multi-way BP** simultaneously generates multiple models with different depth. Take [CIFAR10-MwResNet-56-5](https://disk.yandex.com/d/k1_34p-qvjdCT "MwResNet-56-5") (including the ''auxiliary outputs'' file) for example:

| Intermediate models | Depth | #Params |
| ------------- |:-------------:|:-----:|
|model-15| 15 | 0.03M |
|model-25| 25 | 0.09M |
|model-35| 35 | 0.18M |
|model-45| 45 | 0.48M |
|model-56| 56 | 0.85M |

To test the intermediate models, simply run the script [intermediate.lua](https://github.com/tanmingkui/multiwaybp/tree/master/multiwaybp-torch/intermediate.lua "intermediate").

```
th intermediate.lua -dataset cifar10 -model cifar10-mwresnet-56-5 -outputs cifar10-mwresnet-56-5-outputs
```
