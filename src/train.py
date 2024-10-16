import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Define models to train
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42, probability=True)
}

# Train each model and log results with MLflow
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log model and metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} Accuracy:", accuracy)
