Parkinson's Disease Prediction using Machine Learning
This project aims to detect Parkinson's disease in individuals by leveraging machine learning. The model is built using an XGBoost Classifier and trained on a dataset of biomedical voice measurements.

// Project Overview
The primary goal is to create a reliable predictive model that can distinguish between healthy individuals and those with Parkinson's disease based on specific vocal features. This notebook walks through the entire process, from data exploration and preprocessing to model training, evaluation, and building a simple predictive system.



📊 Dataset
The project utilizes the Parkinson's Dataset sourced from the UCI Machine Learning Repository.

Description: The dataset is composed of 195 voice recordings from 31 individuals, of which 23 have Parkinson's disease.

Features: It contains 22 biomedical voice measures, such as fundamental frequency variation, shimmer, and noise ratios.

Target Variable: The status column, where 1 indicates the presence of Parkinson's and 0 indicates a healthy individual.

⚙️ Methodology
The machine learning workflow is implemented as follows:

Data Loading and Inspection: The parkinsons.data file is loaded into a pandas DataFrame. The data is inspected to understand its structure, check for missing values, and view statistical summaries.

Data Preprocessing: The features (X) are separated from the target variable (Y).

Train-Test Split: The dataset is split into a training set (80%) and a testing set (20%) to prepare for model training and evaluation.

Model Training: An XGBoost Classifier is trained on the training data.

Model Evaluation: The model's performance is assessed using the accuracy score on both the training and testing data to check for reliability and overfitting.

Predictive System: A simple function is created to take new voice measurement data as input and predict whether the individual has Parkinson's disease.

🚀 Results
The trained model achieved the following performance:

Training Data Accuracy: 99.51%

Testing Data Accuracy: 94.87%

The high accuracy on the test set indicates that the model is effective at diagnosing Parkinson's disease from voice measurements.

🛠️ Technologies Used
Python 3

Jupyter Notebook

Pandas for data manipulation.

NumPy for numerical operations.

Scikit-learn for data splitting and model evaluation.

XGBoost for the classification model.

▶ How to Run
Clone the repository:

git clone [https://github.com/nadinejoma/ML-parkinson.git](https://github.com/nadinejoma/ML-parkinson.git)

Navigate to the project directory:

cd ML-parkinson

Install the required libraries:

pip install numpy pandas scikit-learn xgboost

Launch Jupyter Notebook:

jupyter notebook

Open and run the parkinson ML.ipynb notebook to see the full implementation.
