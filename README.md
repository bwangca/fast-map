# A Parallel Implementation of Computing Mean Average Precision
## Introduction
My work frequently involves training new object detectors and testing existing ones. One thing I hate is that testing mAP is such a hassle. You need to set up the dataset in a certain way, install some third-party tools to parse the annotations, use a python script buried deep in someone else's project or pay a few hundred dollars so you can run some ancient MATLAB code, and it might still not work...

The worst part is when you train an object detector from scratch and you wanna see how its mAP improves after each epoch but find out you can't. Existing mAP evaluation scripts are only meant to be used after the training process is complete. However, it must not feel good to find out that your object detector has an mAP of 0.4 after 70 training epochs. Sure, you can look at the change in loss values, but that can be deceiving.

I've been searching for an mAP implementation that can be plugged into a training loop just like classification accuracy. Despite my effort, I can't find any that can be used on the validation set after each training epoch. I mean, if you really want, you can exit the training loop, execute the mAP evaluation script, and start another training epoch. That is just too inefficient.

I couldn't figure out why there's no official implementation from PyTorch or TensorFlow. Their engineers have no issue solving problems multitudes harder than this so what's the reason? I finally understand once I get to the bottom of mAP.

The number one reason that mAP was implemented in a sequential fashion is that the inputs do not have fixed dimensions. For any two different images, the numbers of ground truth bounding boxes are generally not the same, and the numbers of predicted bounding boxes may not be the same as well. However, parallel computing kernels usually require every sample in a batch to have a fixed length along each dimension.

Once the primary reason was found, I located the stages where inputs can be transformed to have a uniform shape. For example, post-processing techniques such as NMS can reduce the number of predicted bounding boxes. To fix this, these functions need to keep discarded bounding boxes and instead use a binary mask to indicate which boxes should actually be kept. This example tells why a parallel implementation of computing mAP is not a straightforward task, because it may require users to update their code. Luckily, most changes are about adding dummy values to fix the irregular shape problem and using binary masks to keep track of real values. Once the changes are made, evaluation of object detectors become much smoother and faster.

This project provides a complete example that shows every aspect that needs to be adapted to incorporate a CUDA-compatible mAP computation into an object detection training/validation routine. Experiments are done on the PascalVOC 2007 test set using ResNet-18 based CenterNet. Currently, my implementation only works with a single IoU threshold. However, it can be easily extended to work with a list of IoU thresholds by using broadcasting so that COCO-style mAP can be computed.

## Installation
Required python packages are torch, numpy, and cv2. To run the demo, you also need to have Jupyter Notebook. Once you have those installed, run make.sh. This script will download the Pascal VOC 2007 test set and the pretrained model. If for some reason the script doesn't work. You can manually download the data from this [link](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) and the model from this [link](https://drive.google.com/file/d/1ZGyOpyN0ho64pEUZxAsMskK3zprNOAFg/view?usp=sharing). Untar the data and place both the data and the model in the root directory of the project. Your folder should look like this:\
fast-map/VOCdevkit\
         resnet18_pascal.pth\
         README.md\
         datasets.py\
         demo.ipynb\
         make.sh\
         metrics.py\
         msra_resnet.py\
         postprocess.py\
         transforms.py\
