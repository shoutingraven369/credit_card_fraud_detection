# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using XGBoost and Isolation Forest algorithms with a Streamlit web application interface.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project implements a machine learning solution for detecting fraudulent credit card transactions. Credit card fraud is a significant problem in the financial industry, and this project aims to address it by building a classification model that can accurately identify fraudulent transactions while minimizing false positives.

The project uses a hybrid approach combining:
- **XGBoost Classifier**: A powerful gradient boosting algorithm for classification
- **Isolation Forest**: An unsupervised anomaly detection algorithm

By combining predictions from both models, the system achieves high recall on fraudulent transactions while maintaining good precision.

## Dataset

The dataset used in this project is the Credit Card Fraud Detection dataset, which contains transactions made by European cardholders in September 2013.

**Dataset Link**: [Google Drive](https://drive.google.com/file/d/13PeneEf7h8JlguYK6kqge8X1YLE_jsEn/view?usp=drive_link)

### Dataset Characteristics

| Property | Description |
|----------|-------------|
| Total Transactions | 284,807 |
| Fraudulent Transactions | 492 (0.172%) |
| Legitimate Transactions | 284,315 (99.828%) |
| Features | 30 (V1-V28 from PCA, Time, Amount) |
| Target Variable | Class (0 = Legitimate, 1 = Fraud) |

**Note**: The dataset is highly imbalanced, with fraudulent transactions making up only 0.172% of all transactions.

## Features

- **Machine Learning Models**: Trained XGBoost and Isolation Forest models for fraud detection
- **SMOTE Oversampling**: Handles class imbalance by generating synthetic samples of the minority class
- **Feature Scaling**: StandardScaler applied to Time and Amount features
- **Web Interface**: Interactive Streamlit application for real-time fraud prediction
- **Model Persistence**: Trained models saved as pickle files for easy deployment

## Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.x |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Web Framework | Streamlit |
| Model Serialization | Joblib |

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/shoutingraven369/credit_card_fraud_detection.git
   cd credit_card_fraud_detection
   ```

2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit joblib matplotlib seaborn
   ```

3. Download the dataset from the provided Google Drive link and place `creditcard.csv` in the project root directory.

## Usage

### Running the Jupyter Notebook

To explore the data analysis and model training process:

```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

Or open it directly in Google Colab using the badge in the notebook.

### Running the Streamlit Application

To launch the web interface for fraud detection:

```bash
streamlit run app.py
```

The application will open in your default web browser. You can:
- Click "Use a random real transaction" to test the model with a sample from the dataset
- View the prediction result indicating whether the transaction is safe or potentially fraudulent
- See the confidence score for each prediction

## Model Architecture

### Data Preprocessing

1. **Missing Values**: Removed rows with missing values using `dropna()`
2. **Feature Scaling**: Applied StandardScaler to normalize `Time` and `Amount` features
3. **Train-Test Split**: 80% training, 20% testing with random state 42

### Handling Class Imbalance

The dataset is highly imbalanced (only 0.17% fraudulent transactions). SMOTE (Synthetic Minority Over-sampling Technique) is used to generate synthetic samples for the minority class during training.

### Model Training

#### XGBoost Classifier
- Configured with `scale_pos_weight` to handle residual class imbalance
- Evaluation metric: Log Loss
- Trained on SMOTE-resampled data

#### Isolation Forest
- Contamination parameter: 0.0017 (estimated proportion of anomalies)
- Used for unsupervised anomaly detection
- Trained on original (non-resampled) training data

### Ensemble Prediction

The final prediction combines both models:
- If **Isolation Forest** flags as anomaly (-1) OR **XGBoost** predicts fraud (1), the transaction is classified as fraudulent
- Otherwise, the transaction is classified as legitimate

## Model Performance

### Classification Report

```
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      9889
         1.0       0.75      1.00      0.86        33

    accuracy                           1.00      9922
   macro avg       0.88      1.00      0.93      9922
weighted avg       1.00      1.00      1.00      9922
```

### Key Metrics

| Metric | Value |
|--------|-------|
| ROC AUC Score | 0.9999 |
| Fraud Recall | 100% |
| Fraud Precision | 75% |
| Overall Accuracy | 99.89% |

### Confusion Matrix

```
[[9878   11]
 [   0   33]]
```

- **True Negatives**: 9,878 (legitimate transactions correctly identified)
- **False Positives**: 11 (legitimate transactions incorrectly flagged as fraud)
- **False Negatives**: 0 (no missed fraudulent transactions)
- **True Positives**: 33 (all fraudulent transactions correctly identified)

## Project Structure

```
credit_card_fraud_detection/
|
|-- credit_card_fraud_detection.ipynb  # Jupyter notebook with EDA and model training
|-- app.py                              # Streamlit web application
|-- xgb_model.pkl                       # Trained XGBoost model
|-- iso_model.pkl                       # Trained Isolation Forest model
|-- project_report.pdf                  # Detailed project report
|-- README.md                           # Project documentation
|-- creditcard.csv                      # Dataset (download separately)
```

## Future Improvements

- Add support for custom transaction input through the web interface
- Implement additional machine learning algorithms (Random Forest, Neural Networks)
- Add feature importance visualization
- Implement real-time API endpoint for integration with banking systems
- Add batch prediction functionality for processing multiple transactions
- Implement model retraining pipeline with new data
- Add more detailed transaction analysis and explanation of predictions

## License

This project is available for educational and research purposes.

---

**Note**: This project is for demonstration purposes only. In production environments, additional security measures, more extensive testing, and regulatory compliance would be necessary.
