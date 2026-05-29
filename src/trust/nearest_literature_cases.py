import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def nearest_cases(train_df: pd.DataFrame, query_df: pd.DataFrame, feature_cols: list[str], k: int = 5) -> pd.DataFrame:
    numeric = [c for c in feature_cols if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c])]
    categorical = [c for c in feature_cols if c in train_df.columns and c not in numeric]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    X_train = train_df[feature_cols].copy()
    X_query = query_df[feature_cols].copy()
    X_train_t = pre.fit_transform(X_train)
    X_query_t = pre.transform(X_query)
    dist = pairwise_distances(X_query_t, X_train_t)
    idx = dist[0].argsort()[:k]
    result = train_df.iloc[idx].copy()
    result["distance_score"] = dist[0][idx]
    return result
