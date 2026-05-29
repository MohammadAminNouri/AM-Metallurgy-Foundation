import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def get_models(random_state: int = 42):
    models = {
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(n_estimators=300, random_state=random_state, objective="reg:squarederror")
    return models


def benchmark_models(df: pd.DataFrame, target: str, feature_cols: list[str], n_splits: int = 5) -> pd.DataFrame:
    data = df.dropna(subset=[target]).copy()
    X = data[feature_cols]
    y = data[target]
    cv = KFold(n_splits=min(n_splits, len(data)), shuffle=True, random_state=42)
    rows = []
    for name, model in get_models().items():
        pipe = Pipeline([("pre", make_preprocessor(X)), ("model", model)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=["r2", "neg_mean_absolute_error"], error_score="raise")
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
    X = data[feature_cols]
    y = data[target]
    model = get_models()[model_name]
    pipe = Pipeline([("pre", make_preprocessor(X)), ("model", model)])
    pipe.fit(X, y)
    return pipe
