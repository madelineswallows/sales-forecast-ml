import argparse
import numpy as np
import pandas as pd
from joblib import dump, load
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


# -----------------------------
# Helpers
# -----------------------------
def _load_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    return df.set_index("date")["sales"]


def _build_feature_frame(y: pd.Series, lags=(7, 28)) -> pd.DataFrame:
    """
    Build features for supervised learning on a *historical* series y.
    Drops rows that don't have all lag/rolling values.
    """
    df = pd.DataFrame({"sales": y})
    # time index (trend proxy)
    df["t"] = np.arange(len(df))

    # lag features
    for L in lags:
        df[f"lag{L}"] = df["sales"].shift(L)

    # rolling means (use prior values only)
    df["roll7"] = df["sales"].shift(1).rolling(7).mean()
    df["roll28"] = df["sales"].shift(1).rolling(28).mean()

    df = df.dropna().copy()
    feat_cols = ["t"] + [f"lag{L}" for L in lags] + ["roll7", "roll28"]
    return df, feat_cols


def _make_next_feature_row(series: pd.Series, t_next: int, lags=(7, 28)) -> pd.DataFrame:
    """
    Build a single feature row for the next step using the current series
    (which may already include prior predictions).
    """
    row = {"t": t_next}
    for L in lags:
        row[f"lag{L}"] = series.iloc[-L] if len(series) >= L else np.nan

    # rolling means using existing last values
    row["roll7"] = series.iloc[-7:].mean() if len(series) >= 7 else series.mean()
    row["roll28"] = series.iloc[-28:].mean() if len(series) >= 28 else series.mean()

    return pd.DataFrame([row])


# -----------------------------
# Fit
# -----------------------------
def fit(data: str, model: str, out: str):
    y = _load_series(data)

    if model == "sarimax":
        sar = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
        res = sar.fit(disp=False)
        dump({"model_type": "sarimax", "model": res}, out)
    else:
        # Build lag/rolling features + time index
        lags = (7, 28)
        df_feat, feat_cols = _build_feature_frame(y, lags=lags)
        X = df_feat[feat_cols].values
        target = df_feat["sales"].values

        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
        )
        xgb.fit(X, target)

        # Save model + metadata needed for forecasting
        dump(
            {
                "model_type": "xgb",
                "model": xgb,
                "feat_cols": feat_cols,
                "lags": list(lags),
            },
            out,
        )

    print(f"Saved model to {out}")


# -----------------------------
# Predict
# -----------------------------
def predict(model_path: str, steps: int, last_date: str, out: str, data_path: str = "data/raw.csv"):
    bundle = load(model_path)
    model_type = bundle["model_type"]

    # Build future date index
    future_dates = pd.date_range(pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=steps, freq="D")

    if model_type == "sarimax":
        sar = bundle["model"]
        preds = sar.forecast(steps=steps)
        fc = pd.DataFrame({"date": future_dates, "forecast": preds})
        fc.to_csv(out, index=False)
        print(f"Wrote forecasts to {out}")
        return

    # XGB path: recursive forecasting with lag/rolling features
    xgb = bundle["model"]
    lags = tuple(bundle["lags"])
    feat_cols = bundle["feat_cols"]

    series = _load_series(data_path).copy()
    preds = []

    # Start t at current length of historical series
    t = len(series)

    for _ in range(steps):
        row = _make_next_feature_row(series, t_next=t, lags=lags)
        # If early steps lack lags (short series), fill with current mean
        row = row.fillna(series.mean())
        yhat = float(xgb.predict(row[feat_cols].values)[0])
        preds.append(yhat)
        # append prediction to series for next-step features
        series = pd.concat([series, pd.Series([yhat], index=[series.index[-1] + pd.Timedelta(days=1)])])
        t += 1

    fc = pd.DataFrame({"date": future_dates, "forecast": preds})
    fc.to_csv(out, index=False)
    print(f"Wrote forecasts to {out}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    fitp = sub.add_parser("fit")
    fitp.add_argument("--data", required=True)
    fitp.add_argument("--model", choices=["xgb", "sarimax"], default="xgb")
    fitp.add_argument("--out", required=True)

    predp = sub.add_parser("predict")
    predp.add_argument("--model", required=True)
    predp.add_argument("--steps", type=int, required=True)
    predp.add_argument("--last_date", required=True)
    predp.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.data, args.model, args.out)
    elif args.cmd == "predict":
        predict(args.model, args.steps, args.last_date, args.out)
