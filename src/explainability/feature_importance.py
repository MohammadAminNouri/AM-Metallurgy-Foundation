import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(model, X: pd.DataFrame, y, n_repeats: int = 10) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
