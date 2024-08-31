# 2MAGCN-FS
This repo is the official implementation for "An Efficient Gesture Recognition for HCI Based on Semantics-Guided GCNs and Adaptive Regularization Graphs"

# Introduction

In the embedded system, real-time gesture recognition is crucial to human-computer interaction (HCI). Recently, Graph Convolutional Networks (GCNs) have been applied to inertial measurement unit-based (IMU-based) gesture recognition. However, the disadvantage of these GCN-based methods is that they use very deep networks to capture deep motion features, without considering computational efficiency. In this paper, we propose a shallow GCN as the basic framework to ensure the real-time performance of gesture recognition. To solve the problem that shallow networks have difficulty in capturing deep motion features, we provide hand-crafted sensor/frame position semantic information to guide deep feature extraction. Furthermore, we propose a regularization module named Double-Mask (2MASK) to enhance the network's generalization. Experiments show that the average inference time on raspberry pi 4b is less than 4 ms. Extensive testing on the dataset indicates that the proposed method outperforms previous state-of-the-art (SOTA) methods on multiple metrics. Experiments in the simulation environment show that our method meets the high-precision and real-time requirements for autonomous taxiing of UAVs.




# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

# Train

python train.py -config config/imu_singlesgn.yaml 

# TEST

python train.py -config config/imu_singlesgn.yaml -eval True -pre_trained_model xxx.pt

# Data

Data will be provided by sending an email to dlh@mail.nwpu.edu.cn.

# License

The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (dlh@mail.nwpu.edu.cn).

# Acknowledgement

This repository is built upon FMA-Net, with data processing techniques adapted from SGN.
