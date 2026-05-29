import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def nearest_cases(train_df: pd.DataFrame, query_df: pd.DataFrame, feature_cols: list[str], k: int = 5) -> pd.DataFrame:
    """Return nearest rows after numeric scaling and categorical one-hot encoding.

    Robust to missing values and unseen categories.
    """
    feature_cols = [c for c in feature_cols if c in train_df.columns and c in query_df.columns]
    if not feature_cols or train_df.empty or query_df.empty:
        return pd.DataFrame()

    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical = [c for c in feature_cols if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))

    pre = ColumnTransformer(transformers)
    X_train = train_df[feature_cols].copy()
    X_query = query_df[feature_cols].copy()
    X_train_t = pre.fit_transform(X_train)
    X_query_t = pre.transform(X_query)
    dist = pairwise_distances(X_query_t, X_train_t)
    idx = dist[0].argsort()[: max(1, min(k, len(train_df)))]
    result = train_df.iloc[idx].copy()
    result["distance_score"] = dist[0][idx]
    return result
