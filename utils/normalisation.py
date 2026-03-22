import numpy as np

def normalize_tfr_robust(X, eps=1e-8):
    """
    Робастная нормализация TFR:
    - вместо среднего — медиана
    - вместо std — IQR (q75 - q25)
    - результат приводится к диапазону [0, 1]

    X shape: (N, C, F, T)
    Нормировка считается по train:
    медиана/квантили берутся по N и T → остаются (C, F)
    """

    # Медиана по trial и времени → (1, C, F, 1)
    median = np.median(X, axis=(0, 3), keepdims=True)

    # Квантили для IQR (робастной шкалы)
    q25 = np.percentile(X, 25, axis=(0, 3), keepdims=True)
    q75 = np.percentile(X, 75, axis=(0, 3), keepdims=True)
    iqr = (q75 - q25) + eps

    # Нормировка:
    # при медиана → 0.5, q25→0, q75→1
    X_norm = (X - median) / iqr + 0.5

    # Жёсткое ограничение диапазона
    # X_norm = np.clip(X_norm, 0.0, 1.0)

    return X_norm