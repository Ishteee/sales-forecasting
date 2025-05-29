# Sales Forecasting App

A simple Streamlit web application built with Python that accepts CSV files and performs prediction using machine learning models such as Gradient Descent, Random Forest, and Neural Networks. The app is designed to work with structured tabular data and gives users quick insights through regression models.

## Features

- Upload any CSV file with structured data
- Choose between multiple ML models:
  - Gradient Descent Regression
  - Random Forest Regressor
  - Neural Network Regressor
- Automatically splits data into training and testing sets
- Displays model performance metrics (MAE, RMSE, R²)
- Shows prediction outputs and charts for better interpretation

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** Python
- **ML Models:** scikit-learn, TensorFlow/Keras (for Neural Network)
