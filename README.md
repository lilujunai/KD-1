# KD
knowledge distillation for keti project

# Overview
This repository contains implementation of Knowlegde distillation on Cifar10 and ImageNet

#For Cifar10
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

# Results on cifar10
|    *Teacher*      |   *Student*           | *Teacher acc* |*Student accuracy achieved with KD*|
|:-----------------:|:---------------------:|:-------------:|:---------------------------------:|
| ResNeXt29_32x8d |   EfficientNet B0   |   [92.66%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/ERZ7knGAZ2tJlTzdjiLFN24BBkCvHhfE3JjUxF9OX1Bpjg?e=f9xoBE)    |            [92.20%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EaUA6nEZzwpMgDB2wfyweqMBPGrap8mub9qF90gw6Jx8pw?e=FKR7Pn)              |
|    Resnet152    |   EfficientNet B0   |   [92.41%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EbF-961igiVEmNH8traaEW8B6shscvJ7Sik3L0AxF8YKzA?e=qyrgcN)    |            [92.12%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EU9PVa2PyHBBnfj4x8CoroQBRhTE3fcDWeBcQwNAk6N1OA?e=Knca0m)               |
|   DenseNet40    |   EfficientNet B0   |   [92.26%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/Eacdmd9AOItNkXWOXPv8HkwBG-8uxrKaoeYJoX7m-8Vn0A?e=G9kQ59)    |            [91.63%](https://gisto365-my.sharepoint.com/:u:/g/personal/ooodragon_gm_gist_ac_kr/EfHrvWKX2KVPrG7WXL9UchEBbJlIfR3SMM7nWhTKdCGZiw?e=MVqkVy)               |


