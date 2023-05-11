# Optimizing Object Detection Models for High Performance on Heterogeneous Hardware Accelerators

## Overview

This repository contains code and resources for a project focused on optimizing the performance of object detection models on various hardware accelerators. The main objective is to analyze the performance using profiling tools, and to identify and address performance bottlenecks using optimization techniques such as quantization, pruning, and model compression. The project covers end-to-end processes from data preprocessing, training, optimization, to evaluation.

## Repository Structure

The repository is organized into the following main directories:

data: This directory is intended for storing datasets used in the project. In this project, we use standard object detection datasets such as COCO and Pascal VOC.
models: This directory contains the object detection models used for this project. These include popular models like YOLO, SSD, and Faster R-CNN.
scripts: This directory houses all the Python scripts necessary for running the project. Detailed descriptions of each script are provided below.
optimization_techniques: This directory contains scripts for different optimization techniques. Subdirectories include compression, pruning, and quantization.
profiling_tools: This directory contains scripts and resources related to profiling tools used in the project. These include NVIDIA Nsight Compute and PyTorch Profiler.

## Scripts

###preprocess_data.py
This script is used for preprocessing the datasets for the object detection models. It includes functionality for handling data specific to YOLO, SSD, and Faster R-CNN.

###train.py
This script is used for training the object detection models. It provides functionality for specifying the batch size, the number of workers, loss function, optimizer, and learning rate. It also supports running on both CPU and GPU based on an argument provided by the user.

###utility_scripts.py
This script contains utility functions used across the project. It includes functionality for creating annotation files, saving and loading models, etc.

###profiling_tools.py
This script contains the GPUProfiler class that is used for profiling GPU usage and timing the execution of code blocks.

###evaluate.py
This script is used for evaluating the performance of the models. It provides functionality for computing and reporting metrics such as precision, recall, and F1-score, as well as profiling the inference time.

## Optimization Techniques

###compression.py
This script contains code for applying model compression techniques to the object detection models.

###pruning.py
This script contains code for applying pruning techniques to the object detection models.

###quantization.py
This script contains code for applying quantization techniques to the object detection models.

##How to Use

Please refer to individual script files for detailed instructions on how to use them. In general, the data preprocessing script should be run first to prepare the datasets, followed by the training script to train the models, and then the evaluation script to evaluate the performance of the models.
