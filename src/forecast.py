import argparse, pandas as pd, numpy as np
from joblib import dump, load
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import joblib, numpy as np, pandas as pd

def fit(data, model, out):
    df = pd.read_csv(data, parse_dates=['date'])
    df = df.sort_values('date')
    y = df['sales'].values
    if model == 'sarimax':
        sar = SARIMAX(y, order=(1,1,1), seasonal_order=(0,0,0,0))
        res = sar.fit(disp=False)
        dump(res, out)
    else:
    # Use time index as feature
        X = np.arange(len(y)).reshape(-1, 1)
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42
        )
        xgb.fit(X, y)
        dump(xgb, out)

    print(f"Saved model to {out}")

def predict(model_path, steps, last_date, out):
    model = load(model_path)

    # Generate future dates
    df_dates = pd.date_range(
        pd.to_datetime(last_date) + pd.Timedelta(days=1),
        periods=steps,
        freq='D'
    )

    if isinstance(model, XGBRegressor):
        # FIX: continue time index from end of training
        hist = pd.read_csv('data/raw.csv', parse_dates=['date']).sort_values('date')
        n_train = len(hist)
        X_future = np.arange(n_train, n_train + steps).reshape(-1, 1)
        preds = model.predict(X_future)
    else:
        preds = model.forecast(steps=steps)

    pd.DataFrame({'date': df_dates, 'forecast': preds}).to_csv(out, index=False)
    print(f"Wrote forecasts to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    fitp = sub.add_parser("fit")
    fitp.add_argument("--data", required=True)
    fitp.add_argument("--model", choices=["xgb","sarimax"], default="xgb")
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
