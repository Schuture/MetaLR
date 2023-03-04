# MetaLR

This repository contains the codes for the paper "MetaLR: Meta-tuning of Learning Rates for Transfer Learning in Medical Imaging".

## Introduction

In medical image analysis, we find that model fine-tuning plays a crucial role in adapting medical knowledge to target tasks. We propose a meta-learning-based LR tuner, MetaLR, to make different layers efficiently co-adapt to downstream tasks according to their transferabilities in different domains.

Previous works fix transferable layers in pre-trained models to prevent them from catastrophic forgetting, which is inflexible and labor-expensive to find the optimal scheme. MetaLR learns appropriate LRs for different layers from feedback on model generalization, preventing highly transferable layers from forgetting their medical representation abilities and driving less transferable layers to adapt actively to new domains.

<img src="https://github.com/Schuture/MetaLR/blob/main/Figs/motivation.png" width = "800" height = "400" alt="Motivation for MetaLR" align=center />


## Algorithm

We use online meta-learning to tune layer-wise LRs. We denote the LR and model parameters for the layer $j$ at the iteration $t$ as $\alpha_j^t$ and $\theta_j^t$. The LR scheduling scheme $\alpha = \\{ \alpha_j^t: j=1, ..., d; ~t=1, ..., T\\}$ is what MetaLR wants to learn, affecting which local optimal $\theta^*(\alpha)$ the model parameters $\theta^t = \\{ \theta_j^t: j=1, ..., d\\}$ will converge to. The algorithm iteratively run the following three steps:

(1) At the iteration $t$ of training, a training data batch $\\{(x_i,y_i),i=1,...,n\\}$ and a validation data batch $\\{(x^v_i,y^v_i):i=1,...,n\\}$ are sampled, where n is the size of the batches. First, the parameters of each layer are updated once with the current LR according to the descent direction on training batch.

$$\hat{\theta_j^t}(\alpha_j^t) = \theta_j^t - \alpha_j^t \nabla_{\theta_j} (\frac{1}{n}\sum_{i=1}^n L(\Phi(x_i,\theta_j^t),y_i)),~j=1,...,d.$$

(2) This step of updating aims to get feedback for LR of each layer. After taking derivative of the validation loss *w.r.t.* $\alpha_j^t$, we can utilize the gradient to know how the LR for each layer should be adjusted. So the second step of MetaLR is to move the LRs along the meta objective gradient on the validation data:

$$\alpha_j^{t+1} = \alpha_j^t - \eta \nabla_{\alpha_j} (\frac{1}{n}\sum_{i=1}^n L(\Phi(x_i^v,\hat{\theta_j^t}(\alpha_j^t)),y_i^v)),$$

where $\eta$ is the hyper-LR.

(3) Finally, the updated LRs can be employed to optimize the model parameters through gradient descent truly.

$$\theta_j^{t+1} = \theta_j^t - \alpha_j^{t+1} \nabla_{\theta_j} (\frac{1}{n}\sum_{i=1}^n L(\Phi(x_i,\theta_j^t),y_i)).$$


## Quick Start

To reproduce MetaLR, you need to 

1. Unzip POCUS dataset in *MetaLR_POCUS/data* folder and unzip the USCL pre-trained model in *MetaLR_POCUS/pretrained* folder.

2. Run

```
cd MetaLR_POCUS
python -u train.py --seed 0 --workers 4\
                   --max-epoch 30 --batch-size 128\
                   --lr 0.01 --hyper-lr 0.1
```

## The Learned LR Curves

<img src="https://github.com/Schuture/MetaLR/blob/main/Figs/lr_curves.png" width = "800" height = "250" alt="Motivation for MetaLR" align=center />


## 5. Environment
The code is developed with an Intel(R) Xeon(R) W-2235 CPU @ 3.80GHz and a single NVIDIA RTX A6000 GPU.

The install script *requirements.txt* has been tested on an Ubuntu 20.04.4 LTS system.

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:


## 6. License

Licensed under an MIT license.




