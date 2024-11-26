README
Custom K-Nearest Neighbors (KNN) Implementation

This project implements a custom version of the K-Nearest Neighbors (KNN) algorithm in Python, aimed at understanding the basic principles of how KNN operates. The code has been written in a clear and modular fashion, making it easy to understand the logic behind each step of the algorithm.
Features

    Implements core KNN functionalities from scratch:
        Training: Memorizes the training dataset.
        Prediction: Predicts the class of a given input point.
        Scoring: Computes the accuracy score of the model.
    Comparison to sklearn's built-in KNN implementation to ensure equivalent performance.

Code Structure
Modules Used

    NumPy: For numerical operations and array handling.
    scikit-learn: For dataset splitting and benchmarking.

Workflow

    Data Preparation
        Loads MNIST dataset (mnist_train_small.npy) containing image data for handwritten digits.
        Splits the dataset into training and testing sets using train_test_split.

    Custom KNN Class
        __init__: Sets the number of neighbors (n_neighbors) for the KNN model.
        fit: Stores the training dataset (X, Y) for later use in predictions.
        predict_point: Computes distances from the test point to all training points, identifies the nearest neighbors, and predicts the class based on majority voting.
        predict: Applies predict_point to all test points.
        score: Evaluates the model by comparing predictions to actual labels.

    Testing
        Trains the custom KNN model on the training set.
        Predicts the first 10 test samples and prints the predicted and actual labels.
        Calculates the accuracy score for a subset of test data.

How to Run

    Clone the repository and navigate to the project directory.
    Ensure that mnist_train_small.npy is placed in the same directory as the script.
    Install the required Python packages:

        pip install numpy scikit-learn
        python custom_knn.py



Example Output

    Data Shape:
    Prints the shape of the dataset:
        (20000, 785)
    Predictions: Outputs the predictions for the first 10 test samples:
        [5 0 4 1 9 2 1 3 1 4]
    Actual Labels:Outputs the actual labels for the first 10 test samples:    
        [5 0 4 1 9 2 1 3 1 4]
    Model Score: Computes the model's accuracy on the test set:
        1.0
Comparison with sklearn's KNN

The custom KNN implementation provides the same predictions and accuracy score as sklearn's built-in KNN when tested with identical hyperparameters and datasets. This verifies the correctness of the implementation.
Future Enhancements

    Optimize distance calculations for larger datasets using efficient data structures (e.g., KD-Trees).
    Add support for different distance metrics (e.g., Manhattan, Cosine).
    Parallelize computation to improve performance on large datasets.

Author

Amr Shafek
Email: amrshafek55@gmail.com
