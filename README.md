# 2MAGCN-FN
This repo is the official implementation for "An Efficient Gesture Recognition for HCI Based on Semantics-Guided GCNs and Adaptive Regularization Graphs"

Note:  We also provide a strong and lightweight model, which shows that the average inference time on raspberry pi 4b is less than 4 ms. Extensive testing on the self-construct dataset indicates that the proposed method outperforms previous state-of-the-art (SOTA) methods on multiple metrics. The accuracy reached 89.47\% on public dataset, leading other methods. Experiments in the simulation environment show that our method meets the high-precision and real-time requirements for autonomous taxiing of UAVs.
## Architecture of 2MAGCN-FN
![image](src/pic3.png)

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- Windows 10, cuda 11.1
- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running pip install -r requirements.txt 

# Data Preparation



1.Request dataset here: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities


2.Download the skeleton-only datasets:


   Ⅰ.daily+and +sports+activities.zip (DSADS)

   Ⅱ.Extract above files to daily+and +sports+activities

# Data Processing

## Directory Structure

```
- daily+and +sports+activities/
  - ddata/
    - a01
    - a02
      ...
    - a19 # raw data of DSADS

```

## Generating Data

- Generate train and test dataset. In this study, we use $p_1$, $p_2$, $p_3$, $p_4$, $p_5$, $p_6$, and $p_7$ as the training set, while $p_8$ serves as the test set.

```
 # Modify the input and output data paths on lines 46 to 48.
 cd ./UCIdata
 # Run the Python script to generate the training and testing datasets.
 # This  Python script will output train.npy train_label.pkl test.npy test_label.pkl
 python UCI.py
```



# Training & Testing


## Train

- Change the config file depending on what you want. (The configuration file is located in config/UCI.yaml.)

```python train.py -config config/UCI.yaml ```

## Pretrained Models

Download pretrained models for producing the final results [[Google Drive]](https://drive.google.com/drive/folders/1YI-4TdKMhfesqc1alfhbV0POiQqKf9A3?usp=sharing).

## TEST

- To test the trained models saved in <model_saved_name>, run the following command:

```python train.py -config config/UCI.yaml -eval True -pre_trained_model xxx.state ```


# License

The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission from the principal investigator (dlh@mail.nwpu.edu.cn).

# Acknowledgement

This repository is built upon SGN.
