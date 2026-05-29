import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers)


def _try_make_xgboost(random_state: int = 42):
    """Lazy optional import so Streamlit still loads when xgboost is absent."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=200,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
        )
    except Exception:
        return None


def get_models(random_state: int = 42, include_optional: bool = False):
    models = {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }
    if include_optional:
        xgb = _try_make_xgboost(random_state)
        if xgb is not None:
            models["xgboost"] = xgb
    return models


def benchmark_models(df: pd.DataFrame, target: str, feature_cols: list[str], n_splits: int = 5, include_optional: bool = False) -> pd.DataFrame:
    data = df.dropna(subset=[target]).copy()
    feature_cols = [c for c in feature_cols if c in data.columns]
    if len(data) < 3:
        raise ValueError("Need at least 3 labelled rows for a meaningful benchmark.")
    if not feature_cols:
        raise ValueError("No usable feature columns were provided.")

    X = data[feature_cols]
    y = data[target]
    cv = KFold(n_splits=min(n_splits, len(data)), shuffle=True, random_state=42)
    rows = []
    for name, model in get_models(include_optional=include_optional).items():
        pipe = Pipeline([("pre", make_preprocessor(X)), ("model", model)])
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=["r2", "neg_mean_absolute_error"],
            error_score="raise",
        )
        rows.append({
            "model": name,
            "target": target,
            "r2_mean": scores["test_r2"].mean(),
            "r2_std": scores["test_r2"].std(),
            "mae_mean": -scores["test_neg_mean_absolute_error"].mean(),
            "mae_std": scores["test_neg_mean_absolute_error"].std(),
            "n_rows": len(data),
        })
    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False)


def train_best_model(df: pd.DataFrame, target: str, feature_cols: list[str], model_name: str = "random_forest"):
    data = df.dropna(subset=[target]).copy()
    feature_cols = [c for c in feature_cols if c in data.columns]
    if len(data) < 2:
        raise ValueError("Need at least 2 labelled rows to train.")
    if not feature_cols:
        raise ValueError("No usable feature columns were provided.")

    X = data[feature_cols]
    y = data[target]
    models = get_models(include_optional=True)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(models)}")
    pipe = Pipeline([("pre", make_preprocessor(X)), ("model", models[model_name])])
    pipe.fit(X, y)
    return pipe
