# Heart Disease Prediction using Machine Learning

This project demonstrates how to build a **Logistic Regression** model to predict the presence of heart disease based on clinical parameters. The project is implemented in **Python** using **scikit-learn**, **pandas**, and **NumPy**.

## 📂 Dataset

The dataset is a CSV file containing medical attributes such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia
- **Target** (1: Heart disease present, 0: No heart disease)

You can find the dataset here:
Data Folder

## 🚀 How to Run

1. **Install Dependencies**

   Make sure you have Python installed (>=3.6). Then install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn


2. Run the Script

Execute the script:

python heart_disease_prediction.py


3. Interpret the Output

The script will:

    Show dataset information

    Train a logistic regression model

    Output training and test accuracy

    Make a prediction on a sample input:

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)


📈 Model

    Model Used: Logistic Regression

    Train/Test Split: 80% training, 20% testing

    Stratified Sampling: Preserves target label distribution.

✨ Example Output
Accuracy on Training data : 0.85
Accuracy on Test data : 0.81
Prediction: [1]
The Person has Heart Disease


📝 Notes

    Ensure your dataset has no missing values.

    You can change the input_data tuple to test other patient scenarios.

    For deployment, consider serializing the model (e.g., with joblib) and building a web API.

📄 License

This project is for educational purposes.

