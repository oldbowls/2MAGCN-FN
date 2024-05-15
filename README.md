# 2MAGCN-FS

#Train
python train.py -config config/imu_singlesgn.yaml 

#TEST

python train.py -config config/imu_singlesgn.yaml -eval True -pre_trained_model xxx.pt
