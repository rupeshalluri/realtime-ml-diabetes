import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import joblib

# Load sample data
from sklearn.datasets import load_diabetes
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model for Docker later
joblib.dump(model, '/dbfs/tmp/model.pkl')

# Log with MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "diabetes_model")
    mlflow.log_metric("r2", model.score(X, y))
