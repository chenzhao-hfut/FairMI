# FairMI
This repository contains source code to our AAAI-2023 paper:

- Fair Representation Learning for Recommendation: A Mutual Information-Based Perspective

FairMI is a framework based Mutual Information (MI) for embedding fairness in recommendations,
which is optimized by the two-fold MI based objective.



>  **Note**:  Due to the limitations of github's file transfer, the complete training parameters are published in https://drive.google.com/drive/folders/1Xe3FSSlyYCfCbJvt0baLjILXdwhHZCAw?usp=sharing



## Getting Started

### Train & Test

- Training BPRMF baseline on MovieLens: 

```shell
python train_bpr_baseline.py
```

- Training FairMI_BPR on MovieLens: 

```shell
python train_bpr_fairmi.py
```

- Training GCN baseline on MovieLens: 

```shell
python train_gcn_baseline.py
```

- Training FairMI_GCN on MovieLens: 

```shell
python train_gcn_fairmi.py
```