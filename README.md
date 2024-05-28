# AI Programming with Python Project
# Image Classifier using PyTorch

## Overview

This project focuses on building and training an image classifier using PyTorch. The classifier is trained on a dataset of flowers to predict the species of flowers in images. The project utilizes a pre-trained VGG16 model for feature extraction and trains a new classifier on top of it.

## Requirements

- Python 3.x
- PyTorch 2.0.1
- torchsummary
- torchvision
- Matplotlib
- NumPy
- PIL (Python Imaging Library)

Ensure you have installed the required dependencies before running the code.

## Workspace Setup

Before proceeding with the code execution, ensure to prepare your workspace:

1. Update the PATH environment variable:

    ```python
    import os
    os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
    os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"
    ```

2. Restart the kernel to apply the PATH changes.

3. Install PyTorch version 2.0.1:

    ```bash
    !python -m pip install torch==2.0.1
    ```

## Usage

1. Load and preprocess the image dataset:

    The dataset consists of training, validation, and testing sets of flower images. Ensure to preprocess the images appropriately before training.

2. Build and train the classifier:

    Utilize a pre-trained VGG16 model for feature extraction and define a new classifier on top of it. Train the classifier layers using backpropagation.

3. Save the trained model checkpoint:

    Once the training is complete, save the trained model along with necessary information such as class-to-index mapping and optimizer state for future use.

4. Load the trained model checkpoint:

    You can load the saved model checkpoint later to make predictions or continue training.

5. Testing the network:

    Evaluate the performance of the trained network on the test dataset to measure its accuracy.

6. Inference for classification:

    Utilize the trained model to predict the class of an input image. Provide the top k most likely classes along with their probabilities.

## Acknowledgments

- This project is part of the "AI Programming with Python" course offered by Udacity.
- PyTorch for providing a powerful deep learning framework.

---
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.
