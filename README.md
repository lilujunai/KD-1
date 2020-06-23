# KD
knowledge distillation for keti project

# Overview
This repository contains implementation of Knowlegde distillation on Cifar10 and ImageNet

# For Cifar10
all available models are in "cifar10/models"
* Vanilla KD is implemented 
* You can change teacher and student pair by changing following line of code

```python
if args.kd:
    net = EfficientNetB0()
    t_net = ResNeXt29_2x64d()
```
* you can also specify a model to run CE on your preferred model

```python
else:
    # net = SPECIFY YOUR MODEL (REFER TO CODE ON cifar10/models)
```

* to run code with CE, simply type
```
bash trainCE.sh
```

* if you want to run code with KD, simply type
```
bash trainKD.sh
```

note that models should be changed manually by hand

For student network, we used EfficientNet-B0 ([90.67%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EfOxqIMI54hJliVkvivB87IB4mRZTF4KoJUw0OtMhn93pQ?e=EE2XkZ)) which is then it was finetuned with KD.

# Results on cifar10

All students' accuracies finetuned with teacher networks were increased and fell in the bound within 1% accuracy drop from that of teachers'.

|    *Teacher*      |   *Student*           | *Teacher acc* |*Student accuracy achieved with KD*|
|:-----------------:|:---------------------:|:-------------:|:---------------------------------:|
| ResNeXt29_32x8d |   EfficientNet B0   |   [92.66%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/ERZ7knGAZ2tJlTzdjiLFN24BBkCvHhfE3JjUxF9OX1Bpjg?e=f9xoBE)    |            [92.20%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EaUA6nEZzwpMgDB2wfyweqMBPGrap8mub9qF90gw6Jx8pw?e=FKR7Pn)              |
|    Resnet152    |   EfficientNet B0   |   [92.41%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EbF-961igiVEmNH8traaEW8B6shscvJ7Sik3L0AxF8YKzA?e=qyrgcN)    |            [92.12%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EU9PVa2PyHBBnfj4x8CoroQBRhTE3fcDWeBcQwNAk6N1OA?e=Knca0m)               |
|   DenseNet40    |   EfficientNet B0   |   [92.26%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/Eacdmd9AOItNkXWOXPv8HkwBG-8uxrKaoeYJoX7m-8Vn0A?e=G9kQ59)    |            [91.63%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EfHrvWKX2KVPrG7WXL9UchEBbJlIfR3SMM7nWhTKdCGZiw?e=MVqkVy)               |


# For ImageNet

Since ResNet152 showed promising accuracy and Size Ratio with EfficientNet-B0, it was chosen as the student and teacher pair for ImageNet

## Size ratio for ResNet152 and EfficientNet B0 on cifar10
    *Teacher*      |   *Student*           | *Teacher Size* |*Student Size*|*compression ratio*|
|:----------------:|:---------------------:|:-------------:|:-------------:|:-----------------:|
|     ResNet152    |   EfficientNet B0     |      72M      |     6.6M      |       9.16%       |

## Size ratio for ResNet152 and EfficientNet B0 on ImageNet
    *Teacher*      |   *Student*           | *Teacher Size* |*Student Size*|*compression ratio*|
|:----------------:|:---------------------:|:-------------:|:-------------:|:-----------------:|
|     ResNet152    |   EfficientNet B0     |     104.4M    |     20.7M     |       19.83%      |

* to run code, type
```
bash train.sh
```
* some options to give in train.sh
```
--arch : you can manually pick student model (efficientnets, models that are in torchvision.)
--teacher-arch : you can manually pick teacher model. They are all pretrained model. For EfficientNet please refer [here](https://github.com/lukemelas/EfficientNet-PyTorch/), otherwise refer to [torchvision documentation](https://pytorch.org/docs/stable/torchvision/models.html)).
--workers : how many threads are used for multiprocessing of CPU (4~8 suggested)
--kd : to do kd or not (if not, only student will be learned with CE)
--T : temperature parameter used for kd
--w : ratio paramter used for kd
--overhaul : implementation of SOTA for kd. (undergoing)
--save_path "$YOUR FOLDER" : saves models on specified folder

other trivial options are not stated here. Please refer to code
```

# Results on ImageNet

|    *Teacher*      |   *Student*           | *Teacher acc* | *Student acc* |*Student accuracy achieved with KD*|
|:-----------------:|:---------------------:|:-------------:|:---------------------------------:|
|     ResNet152     |   EfficientNet B0   |   [78.31%](https://pytorch.org/docs/stable/torchvision/models.html)  | [76.3%](https://github.com/lukemelas/EfficientNet-PyTorch/) | 77.07% |
