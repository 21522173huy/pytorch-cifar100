# Homework2
Practicing CNNs on Chinese characters dataset

## Description
- For homework 1, i used basic resnet152 from pytorch
- For homwork 2, i modified the final layer, classifier layer, into MLP layer and MoE layer

## Requirements
- python 3.11
- pytorch 2.3.0+cu121

## Installation
```
git clone -b main https://github.com/21522173huy pytorch-cifar100.git
cd pytorch-cifar100
```

## Usage
In the train and inference script below, the 'mode' argument will be represented for which type of model you want to use: 
- normal : basic resnet152
- mlp : MLP classifier
- moe : MoE classifier
### For training

```
python train.py \
--train_folder <your_train_folder> \
--val_folder <your_val_folder> \
--test_folder <your_test_folder> \
--label_path <your_952labels_path> \
--mode normal
```
### For inference
```
python download_weight.py # run this code to download pretrained weight
```

```
python inference.py \
--test_folder <your_test_folder> \
--label_path <your_952labels_path> \
--mode moe
```

## Result 
|  Model | Params | Accuracy |
| -------- | ------- | -------- |
| normal  | 60M |96.67|
| mlp  |  61M|  98.13 |
| moe  |  93M| 96.53 |
