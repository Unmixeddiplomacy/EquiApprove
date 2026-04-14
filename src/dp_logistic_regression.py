import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DPLogisticRegression(BaseEstimator, ClassifierMixin):
    """A simple DP-SGD style logistic regression estimator.

    This estimator is intentionally lightweight so it can be used as a drop-in
    base model for Fairlearn reductions that require fit(X, y, sample_weight).
    """

    def __init__(
        self,
        epsilon=8.0,
        delta=1e-5,
        learning_rate=0.1,
        epochs=300,
        clipping_norm=1.0,
        l2=0.01,
        fit_intercept=True,
        random_state=42,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.clipping_norm = clipping_norm
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _augment_intercept(self, X):
        if not self.fit_intercept:
            return X
        intercept = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([intercept, X])

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        unique = np.unique(y)
        if not np.array_equal(np.sort(unique), np.array([0.0, 1.0])):
            raise ValueError("DPLogisticRegression supports only binary labels {0, 1}.")

        X_aug = self._augment_intercept(X)
        n_samples, n_features = X_aug.shape

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight length must match number of samples")

        # Normalize weights to keep optimization stable when Fairlearn reweights.
        sample_weight = sample_weight / (sample_weight.mean() + 1e-12)

        rng = np.random.default_rng(self.random_state)
        self._weights = np.zeros(n_features, dtype=float)

        # Approximate Gaussian mechanism calibration for per-epoch noisy gradients.
        eps = max(float(self.epsilon), 1e-6)
        delta = min(max(float(self.delta), 1e-12), 1.0 - 1e-12)
        sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / eps
        noise_std = sigma * float(self.clipping_norm) / max(n_samples, 1)

        for _ in range(int(self.epochs)):
            scores = X_aug @ self._weights
            probs = self._sigmoid(scores)
            per_sample_grad = (probs - y)[:, None] * X_aug
            per_sample_grad *= sample_weight[:, None]

            norms = np.linalg.norm(per_sample_grad, axis=1, keepdims=True)
            clip_factors = np.minimum(1.0, self.clipping_norm / (norms + 1e-12))
            clipped_grads = per_sample_grad * clip_factors

            grad = clipped_grads.mean(axis=0)
            if self.l2 > 0:
                grad += self.l2 * self._weights

            grad += rng.normal(loc=0.0, scale=noise_std, size=grad.shape)
            self._weights -= self.learning_rate * grad

        if self.fit_intercept:
            self.intercept_ = np.array([self._weights[0]])
            self.coef_ = self._weights[1:].reshape(1, -1)
        else:
            self.intercept_ = np.array([0.0])
            self.coef_ = self._weights.reshape(1, -1)

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_aug = self._augment_intercept(X)
        probs_one = self._sigmoid(X_aug @ self._weights)
        probs_zero = 1.0 - probs_one
        return np.vstack([probs_zero, probs_one]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
