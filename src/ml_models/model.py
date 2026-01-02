import numpy as np
import scipy.sparse as sp
import mlflow.pyfunc


class TelecomLogisticRegression(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        lr=0.01,
        n_iters=1000,
        fit_intercept=True,
        verbose=False,
        class_weight=None  # <--- добавили параметр
    ):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.class_weight = class_weight  # balanced / dict / None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        n = X.shape[0]
        intercept = sp.csr_matrix(np.ones((n, 1)))
        return sp.hstack([intercept, X], format="csr")

    def _compute_class_weights(self, y):
        """Возвращает вес для каждого объекта."""
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)

        if self.class_weight == "balanced":
            # формула из sklearn:
            # w_i = n_samples / (n_classes * count[y_i])
            unique, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(unique)
            class_w = {cls: n_samples / (n_classes * cnt) 
                       for cls, cnt in zip(unique, counts)}

            return np.array([class_w[label] for label in y], dtype=float)

        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight[label] for label in y], dtype=float)

        raise ValueError("Unsupported class_weight")

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # вычисляем вес для каждого объекта
        sample_weights = self._compute_class_weights(y)

        for i in range(self.n_iters):
            linear = X @ self.weights
            y_pred = self._sigmoid(linear)

            # взвешенный градиент
            # grad = X.T @ (w_i * (y_pred - y)) / n
            residual = (y_pred - y) * sample_weights
            grad = (X.T @ residual) / n_samples

            self.weights -= self.lr * grad

            if self.verbose and i % 100 == 0:
                loss = -np.mean(
                    sample_weights * (
                        y * np.log(y_pred + 1e-9) +
                        (1 - y) * np.log(1 - y_pred + 1e-9)
                    )
                )
                print(f"Iter {i}: loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)

        linear = X @ self.weights
        probs = self._sigmoid(linear)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
