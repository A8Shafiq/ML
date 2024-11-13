Handwriting Classification using K-Nearest Neighbors (KNN)

This project is a simple implementation of the K-Nearest Neighbors (KNN) algorithm to classify handwritten digits. The model is trained and evaluated on a subset of the MNIST dataset, a popular dataset for image recognition tasks in machine learning.
Project Overview

The K-Nearest Neighbors (KNN) algorithm is a supervised machine learning algorithm that classifies samples based on their closest labeled neighbors. In this project, we use KNN to classify handwritten digits represented by grayscale images in the MNIST dataset.
Dataset

The model uses the mnist_train_small.npy dataset file, which consists of:

    19,999 samples, each represented as a 785-element vector.
        The first element in each vector is the label (0–9), representing the digit in the image.
        The remaining 784 elements represent pixel values of a 28x28 image.

Code Overview
Key Steps

    Data Loading and Preprocessing: Load the dataset using NumPy and split it into features (x) and labels (y).
    Data Visualization: Display one of the handwritten digits using Matplotlib.
    Train-Test Split: Divide the data into training and testing sets with a 33% test size and a fixed random state (42) for reproducibility.
    Model Training: Initialize and fit a KNN model on the training data.
    Prediction and Evaluation: Predict the first 10 test samples and display the accuracy score for model evaluation.

Code

See knn_handwriting_classification.py for the full code.
Sample Usage

To run the code:

    Ensure mnist_train_small.npy is in the same directory.
    Install the necessary libraries:

    pip install numpy matplotlib scikit-learn

    Execute the script to see results for predictions and model performance.

Example Output

    The first 10 predictions and actual labels from the test set.
    Visualization of the first test image with its predicted label.
    Accuracy score of the KNN model on the test set.

Requirements

    Python 3.x
    Libraries: NumPy, Matplotlib, scikit-learn

Results

The KNN model gives an accuracy score based on the test data. This simple KNN model demonstrates effective digit recognition on a basic handwritten dataset.
