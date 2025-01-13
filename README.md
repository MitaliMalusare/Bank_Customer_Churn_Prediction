# Bank Churn Prediction Project

This repository contains a Bank Churn Prediction project implemented using **ZenML** and **MLflow**. The project demonstrates a machine learning pipeline for predicting customer churn in the banking sector and tracking experiments effectively.

---

## Features

- **End-to-End Pipeline:** Built using ZenML to automate data preprocessing, model training, and evaluation.
- **Experiment Tracking:** MLflow is used to log metrics, parameters, and artifacts.
- **Scalable Architecture:** The project structure supports easy deployment and scalability.
- **Model Evaluation:** Includes performance metrics like accuracy, precision, recall, and F1 score.

---


## Project Structure

```
.
├── analysis/               # Scripts and notebooks for data analysis
│   ├── analyze_src/        # Source scripts for EDA
│   │   ├── basic_data_inspection.py
│   │   ├── bivariate_analysis.py
│   │   ├── missing_values_analysis.py
│   │   ├── multivariate_analysis.py
│   │   └── univariate_analysis.py
│   └── EDA.ipynb           # EDA notebook
├── data/                   # Raw and extracted datasets
│   ├── archive.zip         # Compressed raw data
|-- ├── extracted_data/     # Extracted dataset files
│   └── Bank Customer Churn Prediction.csv
├── mlruns/                 # MLflow tracking artifacts
├── pipelines/              # ZenML pipelines for training and deployment
│   ├── deployment_pipeline.py
│   └── training_pipeline.py
├── src/                    # Core scripts for data preprocessing and modeling
│   ├── data_splitter.py
│   ├── feature_engineering.py
│   ├── handle_missing_values.py
│   ├── ingest_data.py
│   ├── model_building.py
│   ├── model_evaluator.py
│   └── outlier_detection.py
├── steps/                  # ZenML pipeline steps
│   ├── data_ingestion_step.py
│   ├── data_splitter_step.py
│   ├── dynamic_importer.py
│   ├── feature_engineering_step.py
│   ├── handle_missing_values_step.py
│   ├──model_building_step.py
│   ├── model_evaluator_step.py
│   ├── model_loader.py
│   ├── outlier_detection_step.py
│   ├── prediction_service_loader.py
│   └── predictor.py
├── config.yaml         # Configuration file for pipelines
├── run_deployment.py   # Script to run deployment pipeline
├── run_pipeline.py     # Script to run training pipeline
├── sample_predict.py   # Script for sample predictions
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Setup and Installation

### Installation Steps

1. Clone the repository:
   ```bash
   git clone (https://github.com/MitaliMalusare/Bank_Customer_Churn_Prediction.git)
   ```


2. Create a Virtual Environment:
   Run the following command in your project directory:
   ```bash
   python -m venv venv
   ```

   
   Activate the Virtual Environment:

- On **Windows**, run:
  ```bash
  .\venv\Scripts\activate
  ```

- On **MacOS**, run:
  ```bash
  source venv/bin/activate
  ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
---

## How to Run the Project

### Step 1: Run Exploratory Data Analysis

1. Navigate to the `analysis` directory and explore EDA:
   ```bash
   cd analysis
   python analyze_src/univariate_analysis.py
   ```
2. Alternatively, open and execute `EDA.ipynb` in Jupyter Notebook or any compatible IDE.

### Step 2: Configure and Run Pipelines


1. **Install MLflow integration for ZenML:**
   ```bash
   zenml integration install mlflow -y

2. **Register the MLflow model deployer:**
   ```bash
   zenml model-deployer register mlflow --flavor=mlflow

3. **Register and configure the ZenML stack:**
   ```bash
   zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set

4. **Run the training pipeline:**
   ```bash
   python run_pipeline.py
   ```

5. **Run the deployment pipeline:**
   ```bash
   python run_deployment.py
   ```

### Step 3: Make Predictions

Use the `sample_predict.py` script to test predictions:
```bash
python sample_predict.py
```

---

## Results

- Model performance metrics, such as accuracy, precision, recall, and F1 score, are logged in MLflow.
- The trained model and its artifacts are stored in the configured ZenML artifact store.

Model Pipeline Running: 
![image](https://github.com/user-attachments/assets/c8000b46-63fa-4bf2-a84c-6a644d88f615)
![image](https://github.com/user-attachments/assets/152123b1-f3e6-4097-b0fb-94e9ffc5a7e9)


![image](https://github.com/MitaliMalusare/Bank_Customer_Churn_Prediction/blob/main/IMAGES/Capture.JPG)
![image](C:\Users\admin\OneDrive - Vidyalankar Polytechnic\Desktop\ML\ML PROJECTS\predicition\IMAGES\Capture1.JPG)


---

## Dataset

- # Dataset Overview

## 1. Dataset Summary
- **Total Entries**: 10,000  
- **Columns**: 10  
- **Data Types**: 
  - Numerical: All columns are `int64` or `float64`, except for the target variable `churn`, which is binary.

## 2. Summary Statistics

### Numerical Features:

- **Credit Score**:
  - **Mean**: 650.53  
  - **Standard Deviation**: 96.65  
  - **Range**: 350 to 850 (spanning poor to excellent credit scores).  

- **Age**:
  - **Mean**: 38.92 years  
  - **Range**: 18 to 92 years (wide demographic spread).  

- **Tenure**:
  - **Mean**: 5 years  
  - **Range**: 0 to 10 years  
  - **Median**: 5 years (indicates relatively stable customers).  

- **Balance**:
  - **Mean**: 76,485.89  
  - **Standard Deviation**: 62,397.40  
  - **Minimum**: 0 (some customers don’t maintain a balance).  
  - **Maximum**: 250,898.09 (indicating high-value customers).  

- **Products Number**:
  - Most customers own **1 or 2 products** (median: 1, 75th percentile: 2).  
  - **Maximum**: 4 products (suggesting room for cross-selling opportunities).  

- **Estimated Salary**:
  - **Mean**: 100,090.24  
  - **Range**: 11.58 to 199,992.48  
  - **Standard Deviation**: 57,510.49 (diverse customer income levels).  

### Binary Features:

- **Credit Card**:
  - **70.55%** of customers own a credit card (indicating widespread adoption).  

- **Active Member**:
  - **51.51%** of customers are active members (balanced split).  

- **Churn**:
  - **Churn Rate**: 20.37% (approximately 1 in 5 customers leave the bank).  
---


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

---


## Acknowledgements

- [ZenML Documentation](https://docs.zenml.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Scikit-learn](https://scikit-learn.org/)

---

Feel free to reach out for any questions or issues!
