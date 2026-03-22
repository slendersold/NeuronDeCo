import numpy as np

def slope_constraint(trial):
    """
    Optuna ожидает, что constraints_func вернёт последовательность (list/tuple) чисел.
    Никогда не возвращаем None.
    """
    try:
        fold_curves = trial.user_attrs.get("fold_curves", None)
        if not fold_curves:
            # нет данных — считаем “без ограничений”
            return (0.0,)

        # берём val_losses с каждого фолда и считаем slope по хвосту
        slopes = []
        for fc in fold_curves:
            val_losses = fc.get("val_losses", [])
            if not val_losses:
                continue
            tail = val_losses[-10:] if len(val_losses) >= 10 else val_losses
            e = np.arange(len(tail), dtype=np.float64)
            a = float(np.polyfit(e, np.array(tail, dtype=np.float64), 1)[0])
            slopes.append(a)

        if not slopes:
            return (0.0,)

        # Ограничение: хотим slope <= 0, значит constraint = slope (feasible если <=0)
        return (float(np.median(slopes)),)

    except Exception as e:
        # Никогда не падаем — фиксируем причину внутрь trial и считаем 0.0
        trial.set_user_attr("constraint_error", repr(e))
        return (0.0,)