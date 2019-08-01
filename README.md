# SSHA-TensorFlow

### Introduction

This is a implementation of [SSH: Single Stage Headless Face Detector](https://arxiv.org/pdf/1708.03979.pdf) and face keypoints localization reproduced using TensorFlow. 

This code is modified from [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn).	

### Prerequisites

1. You need a CUDA-compatible GPU to train the model.
2. You should first download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) for face detection.

### Dependencies

* TensorFlow 1.4.1
* TF-Slim
* Python3.6
* Ubuntu 16.04
* Cuda 8.0

### Result

demo_result(VGG16_SSHA result):
<div align=center><img src="https://github.com/wanjinchang/SSHA-TensorFlow/blob/master/data/vgg16_result/56.jpg"/></div>
<div align=center><img src="https://github.com/wanjinchang/SSHA-TensorFlow/blob/master/data/vgg16_result/24.jpg"/></div>
<div align=center><img src="https://github.com/wanjinchang/SSHA-TensorFlow/blob/master/data/vgg16_result/45.jpg"/></div>
<div align=center><img src="https://github.com/wanjinchang/SSHA-TensorFlow/blob/master/data/vgg16_result/33.jpg"/></div>

**Result on FDDB**

VGG16-SSH:
<div align=center><img src="https://github.com/wanjinchang/SSHA-TensorFlow/blob/master/data/FDDB_result/VGG16_SSHA_DiscROC.png"/></div>

### Contents

1. [Installation](#installation)
2. [Setup_data](#setup_data)
3. [Training](#training)
4. [Demo](#demo)
5. [Models](#models)

## Installation

-  Clone the repository
  ```Shell
  git clone https://github.com/wanjinchang/SSHA-TensorFlow.git
  ```

-  Update your -arch in setup script to match your GPU
  ```Shell
  cd SSHA-TensorFlow/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | :-------------: | :-------------: |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.

-  Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

## Setup_data

Generate your own annotation file from WIDER FACE dataset(eliminate the invalid data that x <=0 or y <=0 or w <=0 or h <= 0).
    the annotation format looks like follow:  
    image_file_path  
    face_num  
    x_min y_min x_max y_max left_eye_x left_eye_y right_eye_x right_eye_y nose_x nose_y left_mouth_x left_mouth_y right_mouth_x right_mouth_y kpoints_flag  
    ...  
    Here is an example:
```
0--Parade/0_Parade_marchingband_1_849.jpg
1
449.0 330.0 570.0 478.0 488.906 373.643 542.089 376.442 515.031 412.83 485.174 425.893 538.357 431.491 1
0--Parade/0_Parade_Parade_0_904.jpg
1
361.0 98.0 623.0 436.0 424.143 251.656 547.134 232.571 494.121 325.875 453.83 368.286 561.978 342.839 1
0--Parade/0_Parade_marchingband_1_273.jpg
6
178.0 238.0 232.0 310.0 208.679 254.5 217.964 251.714 229.571 266.571 215.179 285.607 218.893 284.679 1
248.0 235.0 306.0 307.0 0 0 0 0 0 0 0 0 0 0 0
363.0 157.0 421.0 229.0 410.0 177.0 416.0 178.0 422.0 188.0 410.0 206.0 416.0 205.0 1
468.0 153.0 520.0 224.0 0 0 0 0 0 0 0 0 0 0 0
629.0 110.0 684.0 190.0 656.5 140.33 684.491 139.821 680.42 156.107 666.679 171.375 686.018 172.393 1
745.0 138.0 799.0 214.0 776.339 155.75 802.857 152.857 797.071 166.357 784.054 183.232 803.339 182.268 1
...
```

**Note**: kpoints_flag: 1 means the face has kpoints annotations, 0 has no kpoints annotations and the kpoints coordinates are all 0.

Or you can use my annotation files `wider_face_train_bbx_kp.txt` and `wider_face_val_bbx_kp.txt` under the folder ``data/`` directly.
And you should have a directory structure as follows:  
```
data
   |--WIDER
         |--WIDER_train
             |--Annotations/
             |--images/ 
         |--WIDER_val
             |--Annotations/
             |--images/ 
```

Or you can follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup WIDER Face datasets. The steps involve downloading data and optionally creating soft links in the ``data/`` folder. 
If you find it useful, the ``data/cache`` folder created on my side is also shared [here](https://drive.google.com/open?id=1L7QpZm5qVgGO8HtDvQbrFcfTIoGY4Jzh).

## Training

-  Download pre-trained models and weights of backbones.The current code supports VGG16/ResNet_V1/MobileNet_Series models. 
-  Pre-trained models are provided by slim, you can get the pre-trained models from [Google Driver](https://drive.google.com/open?id=1iqOZNA9nwvITvwTDvK2gZUHAI1fo_XHI) or [BaiduYun Driver](https://pan.baidu.com/s/1m7uv9Sqs6hEb3VcMy3gFzg). Uzip and place them in the folder ``data/imagenet_weights``. For example, for VGG16 model, you also can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For ResNet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

-  Train
  ```Shell
  ./experiments/scripts/train_ssh.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU id you want to train on
  # NET in {vgg16, res50, res101, res152, mobile, mobile_v2} is the backbone network arch to use
  # DATASET {wider_face} is defined in train_ssh.sh
  # Examples:
  ./experiments/scripts/train_ssh.sh 0 wider_face vgg16
  ./experiments/scripts/train_ssh.sh 1 wider_face res101
  ```
  **Note**: Only support IMS_PER_BATCH=1 for training now, see details in the cfg files under the foder ``experiments/cfgs/``.
 
By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

## Demo

-  For ckpt demo
Download trained models from [Models](#models), then uzip to the folder ``output/``, modify your path of trained model
  ```Shell
  # at repository root
  GPU_ID=0
  CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo_bbox_kpoints.py.
  ```
or run ``python tools/demo_bbox_kpoints.py`` directly.

## Models

* vgg16_ssha(group training models) [BaiduYun Driver](https://pan.baidu.com/s/1hdoQXEZ_NEucGOPiKxQXig, password:45ii)
### License
MIT LICENSE

### TODOs
- [ ] Support multi-batch images training
- [ ] Multi-GPUs training
- [ ] Improve performance

**Note**: Some problems are still under being fixing, and the performance of face detection and face key points localization should be improved...

### References
1. SSH: Single Stage Headless Face Detector(https://arxiv.org/pdf/1708.03979.pdf). Mahyar Najibi, Pouya Samangouei, Rama Chellappa, Larry S. Davis.ICCV 2017.
2. [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
3. [SSH(offical)](https://github.com/mahyarnajibi/SSH)
4. [mxnet-SSHA](https://github.com/templeblock/mxnet-SSHA)
