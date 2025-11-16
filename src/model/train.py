import click
import pandas as pd
import joblib
import os
import mlflow
from mlflow import MlflowClient
from src import init_mlflow
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@click.command()
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("model_output_path", type=click.Path())
@click.argument("n_estimators", type=int, default=100)
def train(train_path: str, model_output_path: str, n_estimators: int):
    """
    Обучение RandomForestClassifier и сохранение модели в MLflow Model Registry (Staging).

    TRAIN_PATH: путь к train CSV
    MODEL_OUTPUT_PATH: путь для сохранения модели
    N_ESTIMATORS: количество деревьев в лесу
    """
    # Инициализация mlflow
    init_mlflow()
    mlflow.set_experiment("churn_experiment")

    df = pd.read_csv(train_path)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # ----------------------------
    # 3. Определяем признаки
    # ----------------------------
    numeric = [
        'account_length', 'number_vmail_messages',
        'total_day_minutes', 'total_day_calls', 'total_day_charge',
        'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
        'total_night_minutes', 'total_night_calls', 'total_night_charge',
        'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
        'number_customer_service_calls'
    ]

    categorical = [
        'state', 'area_code', 'international_plan', 'voice_mail_plan'
    ]

    # ----------------------------
    # 4. Препроцессинг
    # ----------------------------
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric),
            ('cat', categorical_transformer, categorical)
        ]
    )

    # ----------------------------
    # 5. Модель
    # ----------------------------
    classifier = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        max_iter=200
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # ----------------------------
    # 6. MLflow logging
    # ----------------------------
    with mlflow.start_run():

        pipeline.fit(X, y)

        # Параметры модели
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 200)

        # Метрики
        mlflow.log_metric("train_samples", len(df))

        # ------------------------
        # 7. Локальное сохранение
        # ------------------------
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(pipeline, model_output_path)

        # ------------------------
        # 8. Логируем модель в MLflow
        # ------------------------
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="ChurnPipeline"
        )

        client = MlflowClient()

        # Проверка регистрации
        try:
            client.get_registered_model("ChurnPipeline")
        except mlflow.exceptions.RestException:
            client.create_registered_model("ChurnPipeline")

        # Получаем последнюю версию
        versions = client.search_model_versions("name='ChurnPipeline'")
        latest_version = max(int(v.version) for v in versions)

        # Ставим в STAGING
        client.transition_model_version_stage(
            name="ChurnPipeline",
            version=latest_version,
            stage="Staging"
        )

        click.echo(f"✅ Модель ChurnPipeline v{latest_version} загружена в MLflow и отправлена в STAGING")

    click.echo(f"📁 Локально модель сохранена: {model_output_path}")


if __name__ == "__main__":
    train()
