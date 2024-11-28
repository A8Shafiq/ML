Customer Segmentation using K-Means Clustering

This project demonstrates customer segmentation using the K-Means clustering algorithm on a dataset containing customer details from a mall. The objective is to identify distinct customer groups based on their Annual Income and Spending Score.
Project Overview
Steps Involved:

    Data Loading: The dataset is read into a pandas DataFrame from a CSV file (Mall_Customers.csv).
    Data Analysis:
        Display basic information about the dataset.
        Check for null values.
    Feature Selection: Select the relevant features (Annual Income and Spending Score) for clustering.
    WCSS Calculation: Compute the Within-Cluster Sum of Squares (WCSS) for different cluster counts to determine the optimal number of clusters using the Elbow Method.
    K-Means Clustering:
        Apply K-Means with the optimal number of clusters.
        Obtain cluster labels for each customer.
    Visualization: Plot the customer clusters and centroids.

Installation
Prerequisites

Ensure you have Python and the following libraries installed:

    NumPy
    Pandas
    Matplotlib
    Seaborn
    scikit-learn

Install missing libraries using pip:
    pip install numpy pandas matplotlib seaborn scikit-learn

Usage

    Clone this repository or download the script file.
    Place the dataset (Mall_Customers.csv) in the same directory as the script.
    Run the script:
      python customer_segmentation.py

Outputs
1. Elbow Method Graph

This graph helps identify the optimal number of clusters by plotting WCSS against the number of clusters.
2. Cluster Visualization

Displays the customer groups along with their centroids, labeled by their respective clusters.
Dataset

The dataset should have the following columns:

    CustomerID: Unique ID of the customer.
    Gender: Gender of the customer.
    Age: Age of the customer.
    Annual Income: Annual income of the customer in USD.
    Spending Score: Score assigned based on the customer’s spending behavior.

Example Visuals

Elbow Graph


![image](https://github.com/user-attachments/assets/b00cef3f-9650-4dcf-9cb1-8dd7a8dee0eb)

Customer Groups


![image](https://github.com/user-attachments/assets/15287ed7-a0be-40b0-a154-82789755bb8d)



![image](https://github.com/user-attachments/assets/f282359b-c7b0-43c3-ad8c-27b391938e8d)


Author

Amr Shafek
Feel free to reach out for any queries or suggestions!


    
