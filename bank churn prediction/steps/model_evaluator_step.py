import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
from zenml import step
from typing import Tuple
import logging

@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
)-> Tuple[dict, float]:
    """
    Evaluates the trained model using ModelEvaluator and ClassificationModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying the same preprocessing to the test data.")

    # Apply the preprocessing and model prediction
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # Initialize the evaluator with the classification strategy
    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())

    # Perform the evaluation
    evaluation_metrics = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )

    # Ensure that the evaluation metrics are returned as a dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    
    # Extracting individual metrics for clarity
    recall = evaluation_metrics.get("Recall", None)
    accuracy = evaluation_metrics.get("Accuracy", None)
    f1 = evaluation_metrics.get("F1-Score", None)

    # Package the metrics into a more comprehensive dictionary
    additional_metrics = {
        "Recall": recall,
        "Accuracy": accuracy,
        "F1-Score": f1,
    }

    # Return the additional classification metrics
    return additional_metrics, recall  # 