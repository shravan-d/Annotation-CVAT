import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


setup(
    name="yolo",
    #version=get_version(),
    author="FAIR",
    url="https://github.com/shravan-d/Annotation-CVAT",
    description="Custom Yolov7",
    #packages=find_packages(exclude=("configs", "tests*")) + list(PROJECTS.keys()),
    python_requires=">=3.7",
    install_requires=[
        # Usage: pip install -r requirements.txt

        # Base ----------------------------------------
        "matplotlib>=3.2.2",
        "numpy>=1.18.5",
        "opencv-python>=4.1.1",
        "Pillow>=7.1.2",
        "PyYAML>=5.3.1",
        "requests>=2.23.0",
        "scipy>=1.4.1",
        "tqdm>=4.41.0",
        "protobuf<4.21.3",

        # Logging -------------------------------------
        "tensorboard>=2.4.1",
        # wandb

        # Plotting ------------------------------------
        "pandas>=1.1.4",
        "seaborn>=0.11.0",

        # Export --------------------------------------
        # coremltools>=4.1  # CoreML export
        # onnx>=1.9.0  # ONNX export
        # onnx-simplifier>=0.3.6  # ONNX simplifier
        # scikit-learn==0.19.2  # CoreML quantization
        # tensorflow>=2.4.1  # TFLite export
        # tensorflowjs>=3.9.0  # TF.js export
        # openvino-dev  # OpenVINO export

        # Extras --------------------------------------
        "ipython"  # interactive notebook
        "psutil"  # system utilization
        "thop"  # FLOPs computation
        # albumentations>=1.0.3
        # pycocotools>=2.0  # COCO mAP
        # roboflow

    ]
)