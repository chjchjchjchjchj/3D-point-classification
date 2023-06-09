# 3D-point-classification

## Abstract
The purpose of this experiment report is to investigate and evaluate the application of Graph Convolutional Networks (GCN) in the task of 3D point cloud data classification. Point cloud data is a common three-dimensional representation widely used in the field of computer vision. Traditional methods struggle to handle the complex relationships between points in a point cloud, whereas GCN models can leverage the topological structure of point clouds for feature extraction and classification. In this study, I conducted experiments using the ModelNet40 dataset and employed the K-Nearest Neighbors (KNN) algorithm to establish an adjacency matrix for each class and used GCN as a fundamental building block. I progressively increased the number of layers and modified the network architecture to improve the model. As a result, the accuracy of point cloud classification reached **86.79\%**.


## Datasets
I validate my models on datasets Modelnet40. 
ModelNet40 is a widely used dataset for three-dimensional object recognition and understanding. It was created by researchers at Princeton University and serves as a benchmark for machine learning and computer vision algorithms in the field of 3D object classification.
Download the zip from https://modelnet.cs.princeton.edu/ 
```
unzip modelnet40_ply_hdf5_2048.zip -d data
```


## Installation and Dependencies
* python >= 3.9
* torch >=2.0.0

Other dependencies can be installed using the following command:
```
conda create -n gml python=3.9
pip install -r requirements.txt
```


## Usage

* Run the training script by VanillaGCN
```
python main.py model_name=gcn k=20 batch_size=32 epochs=60 gcn_layers=5 address_overfitting=True exp_name=\'gcn,bz:32,epochs:100,k:20,num_layers:5,address_overfitting:True\'
```
* Run the training script by GCNResnet
```
python main.py model_name=gcnresnet res_num_blocks=6 res_hid=64 dropout=0.1 k=20 batch_size=32 epochs=60 gcn_layers=5 address_overfitting=True exp_name=\'gcnresnet,res_num_blocks:6,res_hid:64,dp:0.9,bz:64 epochs:100,k:20,num_layers:5,address_overfitting:True\'
```
* Run the training script by GCNPolynomial
```
python main.py model_name=gcnpolynomial f_times=6 res_hid=64 dropout=0.0 k=20 batch_size=32 epochs=60 gcn_layers=5 address_overfitting=True exp_name=\'gcnpolynomial,f_times:6,res_hid:64,dp:0.0,bz:64,epochs:60,k:20,num_layers:5,address_overfitting:True\'
```

* Run the evaluation script by VanillaGCN
```
python main.py model_name=gcn eval=True model_path=<check_point_path>
```

## Acknowledgement
This project includes the following third-party scripts:

- [DGCNN](https://github.com/WangYueFt/dgcnn/blob/master/pytorch/) by WangYueFt