# Efficiera Residual Networks

## About
This repository is the official implementation of Efficiera Residual Networks (ERNs).
ERNs is a fully w1a2 image classification model including the input and output layer. ERNs achieves competitive performance compared to the state-of-the-art ultra-low-bit quantized models.

## Ask for arxiv endorsement
We need arxiv endorsement to upload our [paper](https://github.com/LeapMind/ERN/blob/main/ERNs_paper.pdf) to arxiv.
If you have the rights to endorse and feel this work is ready to upload on arxiv, please endorse our work from [endorsement link](https://arxiv.org/auth/endorse?x=T3QFLP).

## Accuracy
ImageNet accuracy of the pretrained models. Note that the accuracy may depend on the environment such as GPU architecture.

|     Model    | Model Size (Mbytes) | Top1 w/o TTA | Top5 w/o TTA| Top1 with TTA | Top5 with TTA|
|:------------|:----------------:|:----:|:----:|:----:|:----:|
| ERNs18-0.75x | 0.98                | 60.7 | 83.6 | 63.6 | 85.5 |
|    ERNs18    | 1.4                 | 62.7 | 84.9 | 65.3 | 86.8 |
|    ERNs34    | 2.6                 | 66.7 | 87.8 | 70.0 | 89.7 |
|    ERNs50    | 3.1                 | 70.3 | 90.0 | 72.5 | 91.3 |
|    ERNs101   | 5.6                 | 72.2 | 91.1 | 73.8 | 92.1 |

## Links
- [Paper](https://github.com/LeapMind/ERN/blob/main/ERNs_paper.pdf)
- [Model Checkpoints](https://drive.google.com/drive/folders/1aCQA7QQlZRQTIlpYGENn42O-b5EX71Lv?usp=drive_link)


## Setup
We provide the library as a python wheel file (`efficiera_residual_networks_library-1.0.0-py3-none-any.whl`).
We confirmed that the evaluation of the models work under the following conditions
- python 3.8.0
- NVIDIA DGX Server Version 4.7.0
- NVIDIA-SMI 515.86.01
- CUDA Version 11.7

```
python3.8 -m venv ern_venv
source ./ern_venv/bin/activate
git clone git@github.com:LeapMind/ERN.git
cd ERN
pip install pip==23.3.2 --upgrade # the installation of the wheel package may fail with version 24.0 and higher.
pip install efficiera_residual_networks_library-1.0.0-py3-none-any.whl
```

To train and evaluate models, ImageNet dataset is required.
`root` parameter in `configs/dataset/imagenet.yaml` needs to be modified.

## Training
The model can be trained from scratch using `train.py`.

```
python train.py +experiment=ERNs18x075_imagenet_best
```

The experiment is managed by [hydra](https://hydra.cc/). The experiment settings are managed in `configs/experiment`.
The best experiment settings for each models are as follows.
- ERNs18x075: `configs/experiment/ERNs18x075_imagenet_best.yaml`
- ERNs18: `configs/experiment/ERNs18_imagenet_best.yaml`
- ERNs34: `configs/experiment/ERNs34_imagenet_best.yaml`
- ERNs50: `configs/experiment/ERNs50_imagenet_best.yaml`
- ERNs101: `configs/experiment/ERNs101_imagenet_best.yaml`

## Evaluation
The trained model can be evaluated using `evaluate.py`

```
python evaluate.py +experiment=ERNs18x075_imagenet_best checkpoint_filepath=</path/to/checkpoint_file.ckpt>
```

### Reproduce best results with TTA
The results of Table 2 in our paper are confirmed with the following commands.
Note that the accuracy may be slightly different depending on the environment such as GPU architecture.

#### ERNs18x075
```
python evaluate.py +experiment=ERNs18x075_imagenet_best checkpoint_filepath=./Efficiera_Residual_Networks_Checkpoints/ern18x075/checkpoints/last.ckpt input_image_sizes="[[308, 308]]" pl_module.tencrop_evaluation=true pl_module.tencrop_size=288 training.batch_size=25
```

#### ERNs18
```
python evaluate.py +experiment=ERNs18_imagenet_best checkpoint_filepath=./Efficiera_Residual_Networks_Checkpoints/ern18/checkpoints/last.ckpt input_image_sizes="[[308, 308]]" pl_module.tencrop_evaluation=true pl_module.tencrop_size=288 training.batch_size=5
```

#### ERNs34
```
python evaluate.py +experiment=ERNs34_imagenet_best checkpoint_filepath=./Efficiera_Residual_Networks_Checkpoints/ern34/checkpoints/last.ckpt input_image_sizes="[[306, 306]]" pl_module.tencrop_evaluation=true pl_module.tencrop_size=288 training.batch_size=25
```

#### ERNs50
```
python evaluate.py +experiment=ERNs50_imagenet_best checkpoint_filepath=./Efficiera_Residual_Networks_Checkpoints/ern50/checkpoints/last.ckpt input_image_sizes="[[308, 308]]" pl_module.tencrop_evaluation=true pl_module.tencrop_size=288 training.batch_size=25
```

#### ERNs101
```
python evaluate.py +experiment=ERNs101_imagenet_best checkpoint_filepath=./Efficiera_Residual_Networks_Checkpoints/ern101/checkpoints/last.ckpt input_image_sizes="[[308, 308]]" pl_module.tencrop_evaluation=true pl_module.tencrop_size=288 training.batch_size=25
```



