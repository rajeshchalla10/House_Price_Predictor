# Housing Price Prediction Application README

## Introduction

This application predicts housing prices in Bengaluru using a Ridge Regression model. It is built using Python, Flask, and scikit-learn. The model is trained on the Bengaluru Housing dataset, which contains various features related to houses like location, area, number of bedrooms, etc.

## Features

- Predicts house prices based on user input.
- Uses a Ridge Regression model for prediction, which handles multicollinearity well. Multicollinearity is a common issue in housing datasets.
- Built with Python, Flask, and scikit-learn.
- Uses the Bengaluru Housing dataset for training.

### Prerequisites

- Python 3.6 or higher.
- Required Python libraries: Flask, scikit-learn, pandas, numpy (install using pip).
  ```bash
  pip install Flask scikit-learn pandas numpy

### Usage
Enter the required features of the house in the form.
Click the 'Predict' button.
The application will display the predicted house price.
