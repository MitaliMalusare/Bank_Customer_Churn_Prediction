import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
    "customer_id": [1, 2],
    "credit_score": [720, 680],
    "age": [30, 45],
    "tenure": [5, 7],
    "balance": [20000.0, 15000.0],
    "products_number": [2, 3],
    "credit_card": [1, 0],
    "active_member": [1, 0],
    "estimated_salary": [50000.0, 70000.0],
}

    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
