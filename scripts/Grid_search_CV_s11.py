import sys
from pathlib import Path

# === Настройка путей проекта ===
project_root = Path("/trinity/home/asma.benachour/notebooks/Pirogov/MNE_playground")

sys.path.append(str(project_root))
print("Added to PYTHONPATH:", project_root)

# === Импорты проекта ===
from utils.analysis_pipeline import (
    load_and_preprocess,
    create_epochs,
    save_epochs,
    plot_epochs_images,
)
from utils.config import ch_to_keep, best_ch_by_power

# === Библиотеки ===
import os
import abc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from scipy import signal, stats
from scipy.signal import stft, welch
from mne.filter import notch_filter
from intervaltree import Interval, IntervalTree
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from omegaconf import OmegaConf
import time

def main():
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


    class FeedforwardClassifier(nn.Module):
        def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers=1,
            dropout=0.3,
            activation='ReLU',
            batchnorm=False,
            weight_init='xavier'
        ):
            # Странно, обычно делают так super(self).__init__()
            super(FeedforwardClassifier, self).__init__()

            # --- активация ---
            if isinstance(activation, str):
                if not hasattr(nn, activation):
                    raise ValueError(f"Unknown activation '{activation}' in torch.nn")
                self.activation_cls = getattr(nn, activation)
                self._nonlinearity_name = activation.lower()
            elif isinstance(activation, type) and issubclass(activation, nn.Module):
                self.activation_cls = activation
                self._nonlinearity_name = 'relu'  # разумная дефолтная подсказка для kaiming
            else:
                raise TypeError("activation must be a str (e.g., 'ReLU') or an nn.Module class (e.g., nn.ReLU)")

            self.batchnorm = batchnorm
            self.dropout = dropout
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            layers = []
            in_dim = input_dim

            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.activation_cls())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim

            self.hidden_layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(hidden_dim, output_dim)


        def forward(self, x, return_hidden=False):
            hidden = self.hidden_layers(x)
            out = self.output_layer(hidden)
            return (out, hidden) if return_hidden else out

    from sklearn.base import BaseEstimator, ClassifierMixin

    class TFRDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self,
                    input_dim=200,
                    hidden_dim=256,
                    output_dim=1,
                    num_layers=1,
                    activation='ReLU',
                    dropout=0.3,
                    batchnorm=True,
                    lr=1e-3,
                    epochs=20,
                    batch_size=32,
                    use_pca=True,
                    pca_var=0.95,
                    device=None):
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            self.activation = activation
            self.dropout = dropout
            self.batchnorm = batchnorm
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.use_pca = use_pca
            self.pca_var = pca_var
            
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.pca = PCA(n_components=self.pca_var)
            self.model = None


        def _flatten(self, X):
            return X.reshape(len(X), -1, order='F')


        def fit(self, X, y):
            X = self._flatten(X)

            # PCA если нужно
            if self.use_pca:
                X = self.pca.fit_transform(X)
            self.input_dim = X.shape[-1]

            # модель
            self.model = FeedforwardClassifier(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                activation=self.activation,
                dropout=self.dropout,
                batchnorm=self.batchnorm
            ).to(self.device)

            ds = TFRDataset(X, y)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            optim_ = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(self.epochs):
                self.model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device).unsqueeze(1)
                    optim_.zero_grad()
                    out = self.model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optim_.step()

            return self


        def predict(self, X):
            X = self._flatten(X)
            if self.pca is not None:
                X = self.pca.transform(X)

            ds = TFRDataset(X, np.zeros(len(X)))
            loader = DataLoader(ds, batch_size=self.batch_size)

            preds = []
            self.model.eval()
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    out = self.model(xb)
                    p = torch.sigmoid(out).cpu().numpy().ravel()
                    preds.extend(p)
            
            return (np.array(preds) >= 0.5).astype(int)


    # tfr = mne.time_frequency.read_tfrs("/trinity/home/asma.benachour/notebooks/Pirogov/MNE_playground/tfr_s11.fif")[0]
    # X = normalize_tfr_robust(tfr.copy().crop(tmin=0.0, tmax=1.0).data)[:, :, :10, 100:-400]

    tfr = mne.time_frequency.read_tfrs("/trinity/home/asma.benachour/notebooks/Pirogov/MNE_playground/tfr_s10.fif")[0]
    X = normalize_tfr_robust(tfr.copy().crop(tmin=0.0, tmax=1.0).data)[:,:,5:30,100:-500]
    y = np.where(tfr.events[:, 2] == 9, 1, 0)
    
    dim = X.shape
    X = np.reshape(X, (dim[0], dim[1]*dim[2]*dim[3]))
    models_grid = {
        "FeedForward": {
            "model": TorchMLPClassifier(epochs=30, lr=1e-3),
            "param_grid": {
                "hidden_dim": [128, 256, 512],
                "num_layers": [1, 10, 50],
                "dropout": [0.1, 0.3],
                "lr": [1e-3, 1e-4]
                # "hidden_dim": [256, 512],
                # "num_layers": [1, 2],
                # "activation": ["ReLU", "LeakyReLU"],
                # "dropout":[0.1, 0.2],
                # "batchnorm": [True]
            }
        },

        "SVC": {
            "model": SVC(probability=True),
            "param_grid": {
                "C": [1, 50],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf"]
            },
            # "param_grid":{"C": [200], "gamma": ["scale"], "kernel": ["rbf"]}
        },

        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [5, 10, 15, 20]
            }
        },

        "LogisticRegression": {
            "model": LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000),
            "param_grid": {
                "C": [1, 50, 100],
                "penalty": ["l1", "l2"]
            }
        },

        "GradientBoosting": {
            "model": HistGradientBoostingClassifier(random_state=42),
            "param_grid": {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_leaf_nodes": [15, 31]
            }
        },

        "KNN": {
            "model": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            }
        },

        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        }
    }

    # Кросс-валидация: одна и та же для всех моделей
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Скоринги: явно говорим, чем считать
    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "accuracy": "accuracy",  # внутри использует accuracy_score
    }

    # У каких моделей нужен скейлер
    scale_models = {"SVC", "LogisticRegression", "KNN"}

    results = {}

    for model_name, model_config in models_grid.items():
        model = model_config["model"]
        param_grid = model_config["param_grid"]

        print(f"=====\nmodel {model_name}\n=====")

        # --- собираем Pipeline ---
        steps = []
        if model_name in scale_models:
            steps.append(("scaler", StandardScaler()))

        # PCA для всех (fit только на train каждого фолда)
        steps.append(("pca", PCA(n_components=0.95, random_state=42)))

        steps.append(("model", model))
        pipeline = Pipeline(steps)

        # --- приводим param_grid к виду model__param ---
        param_grid_full = {f"model__{k}": v for k, v in param_grid.items()}

        print(f"\n▶ GridSearchCV for {model_name}")
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid_full,
            scoring=scoring,
            refit="f1_macro",        # лучшая модель по F1_macro
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        # Обучение + внутренняя КВ
        t_start = time.perf_counter()
        grid_search.fit(np.copy(X), np.copy(y))
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        print(f"\n▶ GridSearchCV for {model_name} has finished in {elapsed:.2f} seconds")

        # Индекс лучшей конфигурации
        best_idx = grid_search.best_index_
        cv_res = grid_search.cv_results_

        # Усреднённые по фолдам метрики для лучшего набора параметров
        acc_mean = cv_res["mean_test_accuracy"][best_idx]
        acc_std  = cv_res["std_test_accuracy"][best_idx]
        f1_mean  = cv_res["mean_test_f1_macro"][best_idx]
        f1_std   = cv_res["std_test_f1_macro"][best_idx]

        results[model_name] = {
            "best_params": grid_search.best_params_,
            "cv_accuracy_mean": acc_mean,
            "cv_accuracy_std": acc_std,
            "cv_f1_macro_mean": f1_mean,
            "cv_f1_macro_std": f1_std,
        }
        print("\n================ Cross-Validation Summary ================\n")
        print(f"◆ {model_name}")
        print(f"   CV ACC: {results[model_name]['cv_accuracy_mean']:.3f} ± {results[model_name]['cv_accuracy_std']:.3f}")
        print(f"   CV F1 : {results[model_name]['cv_f1_macro_mean']:.3f} ± {results[model_name]['cv_f1_macro_std']:.3f}")
        print(f"   Best params: {results[model_name]['best_params']}")
        print()


if __name__ == "__main__":
    main()