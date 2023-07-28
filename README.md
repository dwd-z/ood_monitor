# On the Importance of Gradients for Detecting Distributional Shifts in the Wild

Code is modified from [GradNorm](https://github.com/deeplearning-wisc/gradnorm_ood) offical implementation source code.

## Usage

### 1. Dataset Preparation

#### In-distribution dataset

Please download [CIFAR-10](https://cloud.univ-grenoble-alpes.fr/s/ipgfAwg4Fk4CyPd) and place the ID dataset in
`./dataset/id_data/`.

#### Out-of-distribution dataset

Please download [Textures](), [SVHN](https://cloud.univ-grenoble-alpes.fr/s/oaRAzmedmCxxSgf), [LSUN-C](https://cloud.univ-grenoble-alpes.fr/s/cDnrfzr3zF288xk), [LSUN-R](https://cloud.univ-grenoble-alpes.fr/s/Pa8YEZCJRNKtaCe), [iSUN](https://cloud.univ-grenoble-alpes.fr/s/YfAkELSf6PfiaN2), [Places365](), and put all downloaded OOD datasets into `./dataset/ood_data/`.

### 2. Model Preparation

We omit the process of pre-training a classification model on ImageNet-1k.
For the ease of reproduction, we provide our trained models [cifar10_wrn_normal_standard_epoch_199.pt](https://cloud.univ-grenoble-alpes.fr/s/xbZ4R65j9KGnqiN), [cifar10_wrn_logitnorm_standard_epoch_199.pt](https://cloud.univ-grenoble-alpes.fr/s/CpqQK3YrsQ23xqD)
Please put the downloaded model in `./checkpoints/`.

### 3. OOD Detection Evaluation

To reproduce our OoD detection test results, please run:
```
./scripts/test_cifar10.sh MSP(/ODIN/Energy/GradNorm) Textures(/SVHN/LSUN-C/LSUN-R/iSUN/Places365)
```

#### Note for Mahalanobis
Before testing, make sure you have tuned and saved its hyperparameters first by running:
```
./scripts/tune_mahalanobis.sh
```

## OOD Detection Results

