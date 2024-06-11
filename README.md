# 2MAGCN-FS
This repo is the official implementation for "Double-Masked Adaptive Graph Convolutional Network with Semantic Information for Gesture Recognition and Its Application in Autonomous Deck Taxiing of UAVs"

#Introduction

Autonomous deck taxiing of Unmanned Aerial Vehicles (UAVs) on aircraft carriers has received attention in recent years.  However, existing methods lack high-precision and real-time performance when UAVs respond to emergencies such as sudden obstacles.  For the first time, we propose a high-reliability scheme that utilizes a gesture recognition algorithm to replace pilots in recognizing the gestures of the taxiing director. After recognition, the results are directly mapped into UAV control commands. Notably, an embedded gesture recognition algorithm based on a shallow spatial-temporal graph convolutional network framework is proposed to ensure real-time performance. To address the difficulty of shallow networks in capturing deep spatial-temporal features of gestures, a method combining spatial-temporal semantic modules and adaptive graph convolution (AGC) is suggested. Moreover, we propose the double-mask (2MASK) regularization module, which prevents the overfitting of training data by cutting off or smoothing connections in the adaptive graph using two mask matrices. Experiments demonstrate that the gesture recognition Accuracy has reached 99.20%, with an F1-score of 0.9936 and a Jaccard Similarity Coefficient of 0.9612. The inference time on embedded devices is less than 3ms. Furthermore, the proposed method  outperforms previous state-of-the-art (SOTA) methods. Experiments in the simulation environment show that our method meets the high-precision and real-time requirements for autonomous deck taxiing of UAVs.

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

# Train

python train.py -config config/imu_singlesgn.yaml 

# TEST

python train.py -config config/imu_singlesgn.yaml -eval True -pre_trained_model xxx.pt
