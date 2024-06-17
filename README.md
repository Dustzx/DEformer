# [DEformer: A Doctor’s Diagnosis Experience Enhanced Transformer Model for Automatic Diagnosis](https://github.com/Dustzx/DEformer/blob/main/README.md#DEformer)
The  DEformer: A Doctor’s Diagnosis Experience Enhanced Transformer Model for Automatic Diagnosis is built based on DxFormer's decoder-encoder framework. The repo can be used to reproduce the results in the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197624008339):

## [Overview](https://github.com/Dustzx/DEformer/blob/main/README.md#overview)

In this paper, A Doctor’s Diagnosis Experience Enhanced Transformer Model for Automatic Diagnosis model is proposed to learn more implicit experience of doctors. On the Dxy、MZ-4 and MZ-10 dataset, our model outperforms in the core metrics diagnosis accuracy in lower inquiry rounds from 0.7% to 2.0% compared to the most advanced models. In addition, on the MZ-10 dataset our model's symptom recall rate metric  improve 9.4% compared to the previous state-of-the-art model.

## [Setup](https://github.com/Dustzx/DEformer/blob/main/README.md#setup)

The repo mainly requires the following packages.

- nltk 3.3
- python 3.8
- torch 1.7.0+cu110
- torchvision 0.8.1
- scikit-learn 0.20.0

Full packages are listed in requirements.txt.

## [1. Download data](https://github.com/Dustzx/DEformer/blob/main/README.md#1-Download-data)

The dataset can be downloaded as following links:

- [Dxy dataset](https://github.com/HCPLab-SYSU/Medical_DS)
- [MZ-4 dataset](http://www.sdspeople.fudan.edu.cn/zywei/data/acl2018-mds.zip)
- [MZ-10 dataset](https://github.com/lemuria-wchen/imcs21)

## [2. Preprocess](https://github.com/Dustzx/DEformer/blob/main/README.md#2-Preprocess)

```ptyhon
python preprocess.py
```



## [3. Pre-training](https://github.com/Dustzx/DEformer/blob/main/README.md#3-Pre-training)

```python
python pretrain.py
```

## [4. Training](https://github.com/Dustzx/DEformer/blob/main/README.md#4-Training)

```python
python train.py
```

## [5. Inference](https://github.com/Dustzx/DEformer/blob/main/README.md#4-Inference)

```python
python early_stop.py
```

## [Acknowledgement](https://github.com/Dustzx/DEformer/blob/main/README.md#Acknowledgement)

Many thanks to the open source repositories and libraries to speed up our coding progress.

- DxFormer https://github.com/lemuria-wchen/DxFormer

