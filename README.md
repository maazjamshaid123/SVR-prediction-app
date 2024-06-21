# SVR Prediction Web App

This Streamlit web application uses multiple pre-trained Support Vector Regression (SVR) models to make predictions based on user input.

## Features

- Load and use multiple SVR models for prediction.
- Input values for features through the sidebar.
- Display individual predictions from each model.
- Calculate and display the average prediction and standard deviation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/svr-prediction-app.git
    cd svr-prediction-app
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the pre-trained SVR models (`best_model_svr_0.joblib` to `best_model_svr_19.joblib`) are in the same directory as the app.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open the provided URL in your web browser.

3. Input the values for features A, B, C, and D in the sidebar.

4. View individual model predictions and the aggregate results (average prediction and standard deviation) on the main page.

## Code Overview

```python
import streamlit as st
import joblib
import numpy as np

# Load the saved best models
n_ss_splits = 20
best_models = [joblib.load(f"models/best_model_svr_{i}.joblib") for i in range(n_ss_splits)]

# Function to make predictions using each best model
def make_predictions(input_features):
    return [model.predict(input_features.reshape(1, -1))[0] for model in best_models]

# Streamlit web app
st.title("SVR Prediction")

# Input fields for features
input_features = np.array([st.sidebar.number_input(f"Enter value for {feature}", step=0.01) for feature in ['A', 'B', 'C', 'D']])

# Make predictions using each best model
predictions = make_predictions(input_features)

# Display individual predictions
st.header("Individual Predictions from Each Best Model")
num_columns = 3
with st.container():
    col_width = 12 // num_columns
    for i, prediction in enumerate(predictions):
        if i % num_columns == 0:
            col = st.columns(num_columns)
        col[i % num_columns].info(f"Model {i+1}: {prediction:.2f}")

# Calculate and display average and standard deviation
average_prediction = np.mean(predictions)
std_deviation = np.std(predictions)
st.header("Aggregate Results")
st.error(f"Average Prediction: {average_prediction:.2f}")
st.error(f"Standard Deviation: {std_deviation:.2f}")
