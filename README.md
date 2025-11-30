# Credit Card Fraud Detection System

A comprehensive and interactive web application built using Python and Streamlit for detecting fraudulent credit card transactions in real-time. The system uses machine learning models like Isolation Forest and Random Forest to analyze transaction details, providing instant fraud predictions, model performance metrics, and data visualization & insights.

## Overview

The Credit Card Fraud Detection System allows users to input transaction details and get immediate fraud risk assessments. It leverages pre-trained models for anomaly detection and classification, handles imbalanced data with SMOTE, and includes visualizations for better insights. All models and preprocessors are saved for efficient reuse.

## Features

**Real-Time Fraud Prediction**
- Input transaction details like category, amount, location, and more
- Dual predictions from Isolation Forest (anomaly detection) and Random Forest (supervised classification)
- Fraud probability score with visual alerts

**Model Performance Evaluation**

- Accuracy, confusion matrices, and detailed classification reports
- Compare Isolation Forest vs. Random Forest on test data

**Data Visualization & Insights**

- Fraud distribution pie charts and amount distributions
- Geospatial scatter plots of fraud vs. legitimate transactions
- Feature correlation heatmaps and category-wise fraud rates

## Requirements

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (for SMOTE)
- Seaborn
- Matplotlib
- Pickle (built-in)

### Install required libraries:
```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
```

## Installation
- Clone the Repository
```bash
git clone https://github.com/Poorvajha17/Credit-Card-Fraud-Detection-.git
cd Credit-Card-Fraud-Detection-
```
- Download the Dataset
  - This project uses the Credit Card Transactions Fraud Detection Dataset from Kaggle.
  - Download fraudTrain.csv and fraudTest.csv.
  - Place them in a folder named Credit-Card-Fraud-Detection- inside your project directory (or update the file paths in the code).
- Install Dependencies
- Run the Application
```bash
streamlit run main.py
```


