import numpy as np
from src.eeg.preprocessing.normalization import LeakSafeScaler


def test_scaler_leakage():
    N = 20
    C = 4
    T = 50
    X = np.random.randn(N, C, T)
    # split first half train, second half test
    train = X[:10]
    test = X[10:]
    scaler = LeakSafeScaler()
    scaler.fit(train)
    m_train = scaler.mean_.copy()
    _ = scaler.transform(test)
    # ensure scaler params unchanged
    assert np.allclose(m_train, scaler.mean_)
