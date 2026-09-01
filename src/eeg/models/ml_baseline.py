from typing import Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class SKLearnBaseline:
    """A simple classical ML baseline using flattened EEG features."""

    def __init__(self, random_state: Optional[int] = 42):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=self.random_state))
        ])

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        # flatten channels and time into features
        N = X.shape[0]
        return X.reshape(N, -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xf = self._prepare(X)
        self.pipeline.fit(Xf, y)

    def predict(self, X: np.ndarray):
        Xf = self._prepare(X)
        return self.pipeline.predict(Xf)

    def predict_proba(self, X: np.ndarray):
        Xf = self._prepare(X)
        return self.pipeline.predict_proba(Xf)
