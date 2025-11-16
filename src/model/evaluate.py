import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import mlflow
from src import init_mlflow


@click.command()
@click.argument("test_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
def evaluate(test_path: str, model_path: str):
    """
    Evaluate trained model on test set.

    TEST_PATH: path to test CSV
    MODEL_PATH: path to trained model
    """
    init_mlflow()
    mlflow.set_experiment("churn_experiment")

    # -----------------------
    # 2. Load data
    # -----------------------
    df = pd.read_csv(test_path)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # -----------------------
    # 3. Load trained pipeline
    # -----------------------
    clf = joblib.load(model_path)

    # -----------------------
    # 4. Predict
    # -----------------------
    y_pred = clf.predict(X)

    # Metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label="yes")

    report = classification_report(y, y_pred)

    # -----------------------
    # 5. Log metrics to MLflow
    # -----------------------
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # save report as artifact
        with open("classification_report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("classification_report.txt")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Classification report saved to MLflow.")


if __name__ == "__main__":
    evaluate()
