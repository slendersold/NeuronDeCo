# Раскладка `lib/` (модели + Optuna)

## Модели

- **`lib/models/alexnet/`** — разбор AlexNet-TFR: `backbone.py`, `head.py`, `model.py`.
- **`lib/models/tfr_transformer/`** — код из `notebooks/transformer_17_03_26.ipynb`: препроцессинг, `TFRSequenceTransformer`, `TFRTransformerWrapper`. Логиты классов на каждом шаге времени + `SeqPool` → `[B, num_classes]` (совместимо с `cross_entropy`).

Публичные импорты: `lib.models.AlexNetTFR`, `lib.models.TFRTransformerWrapper`.  
`lib/AlexNet/AlexNet.py` и `utils/AlexNet.py` — тонкие реэкспорты для старых путей импорта.

## Optuna objective engine

Пакет **`lib/optuna/`**:

| Файл | Роль |
|------|------|
| `types.py` | `Split`, `FoldResult`, `Params`, … |
| `metrics.py` | `loss_slope` (устойчив к 1 эпохе / сбою polyfit), `aggregate` |
| `splits.py` | `make_splits_fn_factory` |
| `fold_runner.py` | `run_fold_fn_factory` |
| `params_alexnet.py` | `params_fn_factory` (= AlexNet) |
| `params_transformer.py` | `params_fn_factory_transformer` (+ `seq_len` из `X.shape[3]`) |
| `objectives.py` | `objectives_fn`, `attrs_fn` |
| `engine.py` | `make_objective_engine` |

`lib/optuna_objective_makers.py` оставлен как **`from lib.optuna import *`** для ноутбуков.

## Ноутбук transformer

Вместо дублирования ячеек с классами:

```python
from lib.models.tfr_transformer import TFRTransformerWrapper
from lib.optuna import (
    params_fn_factory_transformer,
    make_objective_engine,
    make_splits_fn_factory,
    run_fold_fn_factory,
    loss_slope,
    objectives_fn,
    attrs_fn,
)
# params_fn = params_fn_factory_transformer(num_classes=2, seq_len=X.shape[3])
```
