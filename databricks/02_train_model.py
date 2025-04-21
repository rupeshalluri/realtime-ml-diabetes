import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

# Load cleaned data
df = pd.read_csv('/dbfs/tmp/diabetes_clean.csv')
X = df.drop('target', axis=1)
y = df['target']

# Start MLflow run
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X, y)
    
    # Log model
    mlflow.sklearn.log_model(model, "diabetes_model")
    mlflow.log_metric("r2_score", model.score(X, y))
