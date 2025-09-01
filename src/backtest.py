import argparse, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

def main(data, freq, horizon, lags, model):
    df = pd.read_csv(data, parse_dates=['date'])
    df = df.sort_values('date')
    y = df['sales'].values
    # naive rolling forecast simulation
    preds, actuals = [], []
    for i in range(lags, len(y)-horizon, horizon):
        train_y = y[:i]
        test_y = y[i:i+horizon]
        if model == 'sarimax':
            sar = SARIMAX(train_y, order=(1,1,1), seasonal_order=(0,0,0,0))
            res = sar.fit(disp=False)
            fc = res.forecast(steps=horizon)
        else:
            X_train = np.arange(len(train_y)).reshape(-1,1)
            xgb = XGBRegressor(n_estimators=50)
            xgb.fit(X_train, train_y)
            X_test = np.arange(len(train_y), len(train_y)+horizon).reshape(-1,1)
            fc = xgb.predict(X_test)
        preds.extend(fc)
        actuals.extend(test_y)
    mae = mean_absolute_error(actuals, preds)
    print("MAE:", mae)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--freq", default="D")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--lags", type=int, default=28)
    ap.add_argument("--model", choices=["xgb","sarimax"], default="xgb")
    args = ap.parse_args()
    main(args.data, args.freq, args.horizon, args.lags, args.model)
