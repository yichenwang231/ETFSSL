# ETFSSL
This is the code for the paper"Neural Collapse Inspired Semi-Supervised Learning with Fixed Classifier"

# Environment
- python==3.8
- pytorch==1.11.1
- torchvision==0.11.2
- CUDA==11.1
- scipy==1.7.3
- Pillow==9.0.0
- PyYAML==6.0
- semilearn==0.3.0
# Usage
ETFSSL is composed of the pretraining and joint training stages. The settings for the parameters of the experiment and datasets are in the config directory.
# Data Preparation
Download the raw dataset of UrbanSound8k from "https://urbansounddataset.weebly.com/urbansound8k.html", and then run ```
python preprocess/preprocess_urbansound.py ``` to process the raw dataset. Other datasets can be downloaded from the url provided by their corresponding papers or official websites.
# Dataset Structure:
Make sure to put the files in the following structure:
```
|-- data
|   |-- cifar100
|   |-- UC-Merced
|   |-- amazon_review
|   |-- ...
```
# Pretraining On Few Labeled Data
After setting the configuration, to start training, simply run
```
python train.py --c config/cv/etfssl_cifar100_200_0.yaml 
```
# Joint Training With Labeled And Unlabeled Data
```
python train.py --c config/cv/etfssl_cifar100_200_0.yaml --mode reload
```
# Acknowledge
ETFSSL is developed based on the architecture of "USB: A Unified Semi-supervised Learning Benchmark for Classification" (NIPS 2022, [https://github.com/Yunfan-Li/Contrastive-Clustering](https://github.com/microsoft/Semi-supervised-learning)).We sincerely thank the authors for the excellent works!
