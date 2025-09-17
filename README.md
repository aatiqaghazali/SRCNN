# SRCNN
This repository contains a Keras/TensorFlow implementation of the SRCNN (Super-Resolution Convolutional Neural Network) proposed by Dong et al., 2014.
SRCNN is one of the first deep learning models for image super-resolution, reconstructing high-resolution images from low-resolution inputs.
# Paper Reference/link:
https://arxiv.org/pdf/1501.00092 : Learning a Deep Convolutional Network for Image Super-Resolution (ECCV 2014)
# Repository Structure
SRCNN/
├── train.py        # Training loop
├── test.py         # Evaluation on test images
├── model.py        # SRCNN model architecture
├── dataset.py      # Data loading + preprocessing
├── utils.py        # Helper functions (PSNR, SSIM, resizing)
└── README.md       # Project documentation
# How to Run
DIV2K dataset is used here:
HR images: DIV2K_train_HR
LR images (bicubic x2): DIV2K_train_LR_bicubic_X2
Validation/Test: DIV2K_valid_HR, DIV2K_valid_LR_bicubic_X2
# Evaluation Metrics
PSNR : 37dB
SSIM : 0.94
# Notes
SRCNN first upsamples LR images using bicubic interpolation, then refines details with CNN. (this step is replicated here in dataset.py file)
patch-based training is done in this code for efficiency instead of full 2K images. (as loading all images in one go will crash RAM)

