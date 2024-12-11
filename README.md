# Medical-Image-Deblurring-MATLAB
A MATLAB implementation for enhancing blurry medical images using a U-Net architecture
Medical Image Deblurring with U-Net (MATLAB)
This repository contains a MATLAB implementation of a deep learning-based method for enhancing blurry medical images using a U-Net architecture. The goal is to demonstrate the use of Deep Image Prior (DIP) techniques for deblurring and enhancing the quality of medical images, such as X-ray scans.

Features
Deep Image Prior (DIP): Utilizes the concept of DIP where the structure of the neural network acts as a prior for image reconstruction tasks.
U-Net Architecture: A convolutional neural network designed for image denoising and enhancement.
Medical Image Focus: Specifically designed for deblurring blurry medical images like X-ray scans.
Requirements
Before running the code, ensure that you have the following installed:

MATLAB (R2020b or later recommended)
Deep Learning Toolbox
Image Processing Toolbox
Code Overview
U-Net Architecture
This implementation uses a simple U-Net architecture which is commonly used for image segmentation and image reconstruction tasks. The U-Net consists of:

Encoder: For extracting features from the input image.
Decoder: For reconstructing the image based on the features extracted.
Forward Operator
The forward operator mimics real-world degradation of images, such as adding blur or noise. In this case, we simulate blurring in medical images.

Inverse Problem Solution
The inverse problem is solved using a combination of U-Net and DIP. The model learns to reconstruct the clear version of a blurred medical image without requiring a pre-trained model.

Usage
Steps to Run
Clone this repository:

https://github.com/YOUR-USERNAME/Medical-Image-Deblurring-MATLAB.git

Make sure to set the path for the input medical image (such as an X-ray) in the code:

image = imread('path_to_your_xray_image');

Run the script in MATLAB:

run('inverse_imaging_problem_solution.m')

The program will process the image using the U-Net model and display the original blurred image along with the enhanced (deblurred) result.

Parameters
Image Path: Modify the image variable to point to your medical image file.
Optimizer Settings: The Adam optimizer is used with a learning rate of 1e-3.
Iterations: The default number of optimization iterations is set to 1000.
Sample Usage
You can run the code for an X-ray image by simply calling the script:

Output
The deblurred image will be displayed along with the original image for comparison.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
The U-Net architecture was inspired by the original paper U-Net: Convolutional Networks for Biomedical Image Segmentation.
The Deep Image Prior (DIP) approach is based on the paper Deep Image Prior.
