import optuna

def pareto_front(trials, directions):
    
    def dominates(a, b) -> bool:
        better_or_equal = True
        strictly_better = False
        for av, bv, d in zip(a.values, b.values, directions):
            if d == optuna.study.StudyDirection.MAXIMIZE:
                if av < bv: better_or_equal = False
                if av > bv: strictly_better = True
            else:  # MINIMIZE
                if av > bv: better_or_equal = False
                if av < bv: strictly_better = True
        return better_or_equal and strictly_better

    front = []
    for t in trials:
        if not any(dominates(o, t) for o in trials if o.number != t.number):
            front.append(t)
    return front

def feasible_trials_less_zero(study):

    return [t for t in study.trials if t.values[1] <= 0]