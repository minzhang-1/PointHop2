# PointHop++: *A Lightweight Learning Model on Point Sets for 3D Classification*
Created by [Min Zhang](https://github.com/minzhang-1), [Yifan Wang](https://github.com/yifan-fanyi), Pranav Kadam, Shan Liu, C.-C. Jay Kuo from University of Southern California.

![introduction](https://github.com/minzhang-1/PointHop2/blob/master/doc/baseline.png)

### Introduction
This work is an official implementation of our [arXiv tech report](https://arxiv.org/abs/2002.03281). We improve the [PointHop method](https://arxiv.org/abs/1907.12766) furthermore in two aspects: 1) reducing its model complexity in terms of the model parameter number and 2) ordering discriminant features automatically based on the cross-entropy criterion. The resulting method is called PointHop++. The first improvement is essential for wearable and mobile computing while the second improvement bridges statistics-based and optimization-based machine learning methodologies. With experiments conducted on the ModelNet40 benchmark dataset, we show that the PointHop++ method performs on par with deep neural network (DNN) solutions and surpasses other unsupervised feature extraction methods.

In this repository, we release code and data for training a PointHop++ classification network on point clouds sampled from 3D shapes.

### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2020pointhop++,
	  title={PointHop++: A Lightweight Learning Model on Point Sets for 3D Classification},
	  author={Zhang, Min and Wang, Yifan and Kadam, Pranav and Liu, Shan and Kuo, C-C Jay},
	  journal={arXiv preprint arXiv:2002.03281},
	  year={2020}
	}

### Installation

The code has been tested with Python 3.5. You may need to install h5py, pytorch, sklearn, pickle and threading packages.

To install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

### Usage
To train a single model without feature selection and ensemble to classify point clouds sampled from 3D shapes:

    python3 train.py

After the above training, we can evaluate the single model. You can also use the provided model `params_single_wo_fe` to do evaluation directly.

    python3 evaluate.py

Log files and network parameters will be saved to `log` folder. If you would like to achieve better performance, you can change the argument `feature_selection` from `None` to `0.95` or `ensemble` from `False` to `True` or both in `train.py` and `evaluate.py` respectively. Or use the provided model `params_single_w_fe` and `params_ensemble_w_fe`. 

Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.


