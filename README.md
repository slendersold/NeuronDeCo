# NeuronDeCo

- `lib/` — ядро: `core/`, `modes/`, `models/`, `experiments/`, пакет **`lib/optuna/`** (движок objective), см. **`docs/lib-module-layout.md`**.
- `examples/` — короткие скрипты.
- `utils/` — датасеты и хелперы обучения, устаревшая часть.

### Примеры

- **Оффлайн + синтетика + AlexNet / Transformer:**  
  `python examples/offline_synthetic_tfr.py`  
  Режим `offline_tfr_supervised` (реальный цикл AdamW + `train_eval_helpers`).

- **Optuna + синтетика** (мультиobjective F1 / slope):  
  `python examples/optuna_synthetic_tfr.py`  
  Явный `study.optimize`; см. `--help`, `--n-trials`, `--alexnet-only` / `--transformer-only`.

- **Скелет с `fit`:** `python examples/minimal_framework_skeleton.py` (нужен sklearn-подобный `fit`; для `nn.Module` используй пример выше или Optuna).
