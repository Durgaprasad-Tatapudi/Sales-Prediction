# Sales-Prediction
 A machine learning model to forecast product sales based on historical data and influencing factors.

## Project Structure
```
Sales-Prediction/
│-- data/
│   ├── raw/
│   │   ├── sales_data.csv
│   ├── processed/
│   │   ├── cleaned_data.csv
│-- notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│-- src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│-- reports/
│   ├── model_performance.pdf
│-- requirements.txt
│-- README.md
```

## Overview
This project aims to develop a machine learning model that predicts sales based on various factors like advertising spend, promotions, and customer segmentation. The model is trained using historical sales data, with preprocessing, feature engineering, and evaluation steps documented.

## Dataset
The dataset consists of historical sales records, including:
- Age
- Annual Salary
- Credit Card Debt
- Net Worth
- Car Purchase Amount (Target Variable)

## Data Preprocessing
- Renamed columns for consistency.
- Handled missing values by filling with the median.
- Encoded categorical variables (e.g., gender).
- Performed feature scaling using normalization.

## Model Selection
A **Linear Regression** model is used due to:
- Its interpretability and efficiency for numerical regression tasks.
- The dataset structure being well-suited for a linear approach.
- Comparatively low computational cost.

## Model Training
- Split data into 80% training and 20% testing sets.
- Trained a **Linear Regression** model on scaled features.
- Used a **mean squared error (MSE) loss function.
## Model Evolution 
Mean Absolute Error
Mean Squared Error
Root Mean Squared Error
R-Squared Score

## Visualizations
- Distribution of car purchase amounts.
- Correlation heatmap of features.
- Actual vs Predicted scatter plot.
- Residual plot to check model errors.

## Results
The model effectively forecasts sales trends and provides insights into the factors influencing purchases. The findings can help optimize marketing strategies and budget allocation for higher sales growth.


