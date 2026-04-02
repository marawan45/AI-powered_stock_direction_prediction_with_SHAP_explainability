"""
StockSense AI — Training Pipeline

Predicts whether a stock will go UP or DOWN in the next N days.

Features engineered from OHLCV data:
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
  - Rolling statistics (momentum, volatility, volume trends)
  - Lag features

Model: XGBoost classifier + SHAP explanations


"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

#  Paths 
BASE_DIR      = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

SEED           = 42
FORWARD_DAYS   = 5       # Predict price direction N days ahead
THRESHOLD_PCT  = 0.015   # +1.5% = BUY signal, else HOLD/SELL
np.random.seed(SEED)

FEATURE_DISPLAY = {
    "rsi_14":            "RSI (14)",
    "macd":              "MACD",
    "macd_signal":       "MACD Signal",
    "macd_hist":         "MACD Histogram",
    "bb_pct":            "Bollinger %B",
    "bb_width":          "Bollinger Width",
    "atr_14":            "ATR (14)",
    "ema_9":             "EMA 9",
    "ema_21":            "EMA 21",
    "ema_cross":         "EMA 9/21 Cross",
    "sma_50":            "SMA 50",
    "price_vs_sma50":    "Price / SMA50",
    "momentum_10":       "Momentum 10d",
    "momentum_20":       "Momentum 20d",
    "roc_5":             "Rate of Change 5d",
    "roc_10":            "Rate of Change 10d",
    "vol_change":        "Volume Change",
    "vol_ma_ratio":      "Volume / MA(20)",
    "volatility_10":     "Volatility 10d",
    "volatility_20":     "Volatility 20d",
    "high_low_pct":      "High-Low %",
    "close_open_pct":    "Close-Open %",
    "lag_ret_1":         "Return Lag 1d",
    "lag_ret_2":         "Return Lag 2d",
    "lag_ret_3":         "Return Lag 3d",
    "lag_ret_5":         "Return Lag 5d",
    "lag_ret_10":        "Return Lag 10d",
}


#  Synthetic OHLCV Data Generator 
def generate_ohlcv(
    n_days: int = 3000,
    ticker: str = "SYNTH",
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate realistic OHLCV price series using geometric Brownian motion
    with regime switching (bull/bear/sideways) and volume spikes.
    """
    rng = np.random.default_rng(seed)

    # Regime-switching drift
    regimes = rng.choice(["bull", "bear", "sideways"], size=n_days,
                          p=[0.40, 0.25, 0.35])
    drift_map = {"bull": 0.0008, "bear": -0.0005, "sideways": 0.0001}
    vol_map   = {"bull": 0.012,  "bear": 0.020,   "sideways": 0.008}

    log_returns = np.array([
        rng.normal(drift_map[r], vol_map[r]) for r in regimes
    ])

    close = 100.0 * np.exp(np.cumsum(log_returns))

    # OHLC from close
    noise = rng.uniform(0.002, 0.015, n_days)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = close * (1 + rng.normal(0, 0.005, n_days))

    # Volume with spikes
    base_vol   = rng.lognormal(mean=14.5, sigma=0.5, size=n_days).astype(int)
    spike_mask = rng.random(n_days) < 0.05
    base_vol[spike_mask] *= rng.integers(2, 6, spike_mask.sum())

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    df = pd.DataFrame({
        "date":   dates,
        "open":   open_.round(4),
        "high":   high.round(4),
        "low":    low.round(4),
        "close":  close.round(4),
        "volume": base_vol,
        "ticker": ticker,
    }).set_index("date")

    return df


#  Feature Engineering 
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd       = ema_fast - ema_slow
    macd_sig   = macd.ewm(span=signal, adjust=False).mean()
    macd_hist  = macd - macd_sig
    return macd, macd_sig, macd_hist


def compute_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma      = series.rolling(period).mean()
    std      = series.rolling(period).std()
    upper    = sma + num_std * std
    lower    = sma - num_std * std
    bb_pct   = (series - lower) / (upper - lower + 1e-9)   # %B
    bb_width = (upper - lower) / (sma + 1e-9)
    return bb_pct, bb_width


def compute_atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator features to the dataframe."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Trend
    df["ema_9"]         = c.ewm(span=9,  adjust=False).mean()
    df["ema_21"]        = c.ewm(span=21, adjust=False).mean()
    df["ema_cross"]     = df["ema_9"] - df["ema_21"]
    df["sma_50"]        = c.rolling(50).mean()
    df["price_vs_sma50"]= c / (df["sma_50"] + 1e-9)

    # Momentum
    df["rsi_14"]        = compute_rsi(c, 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(c)
    df["momentum_10"]   = c / c.shift(10) - 1
    df["momentum_20"]   = c / c.shift(20) - 1
    df["roc_5"]         = c.pct_change(5)
    df["roc_10"]        = c.pct_change(10)

    # Volatility
    df["bb_pct"], df["bb_width"] = compute_bollinger(c)
    df["atr_14"]        = compute_atr(h, l, c)
    df["volatility_10"] = c.pct_change().rolling(10).std()
    df["volatility_20"] = c.pct_change().rolling(20).std()
    df["high_low_pct"]  = (h - l) / (c + 1e-9)
    df["close_open_pct"]= (c - df["open"]) / (df["open"] + 1e-9)

    # Volume
    df["vol_change"]    = v.pct_change()
    df["vol_ma_ratio"]  = v / (v.rolling(20).mean() + 1e-9)

    # Lag returns
    ret = c.pct_change()
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_ret_{lag}"] = ret.shift(lag)

    # Target: will price be >THRESHOLD_PCT higher N days from now?
    df["future_return"] = c.shift(-FORWARD_DAYS) / c - 1
    df["target"]        = (df["future_return"] > THRESHOLD_PCT).astype(int)

    return df


#  Model Pipeline 
def build_dataset(df: pd.DataFrame):
    FEATURES = list(FEATURE_DISPLAY.keys())
    df = df.dropna(subset=FEATURES + ["target"])
    X  = df[FEATURES]
    y  = df["target"]
    return X, y


def plot_evaluation(y_test, y_pred, y_prob, save_dir: Path) -> None:
    auc       = roc_auc_score(y_test, y_prob)
    avg_prec  = average_precision_score(y_test, y_prob)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("StockSense AI — Model Evaluation", fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="#00d4ff", lw=2.5, label=f"AUC={auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--",lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="#00d4ff")
    axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    axes[0].legend(); axes[0].set_title("ROC Curve", fontweight="bold")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, color="#f39c12", lw=2.5, label=f"AP={avg_prec:.3f}")
    axes[1].fill_between(rec, prec, alpha=0.1, color="#f39c12")
    axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend(); axes[1].set_title("Precision-Recall Curve", fontweight="bold")

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    sns.heatmap(cm, ax=axes[2], annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=["DOWN","UP"], yticklabels=["DOWN","UP"])
    axes[2].set(xlabel="Predicted", ylabel="True")
    axes[2].set_title("Confusion Matrix", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_importance(shap_vals, feature_names, save_dir: Path) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "shap": mean_abs})
    imp = imp.sort_values("shap").tail(15)
    imp["display"] = imp["feature"].map(FEATURE_DISPLAY)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp)))
    ax.barh(imp["display"], imp["shap"], color=colors, edgecolor="none", height=0.65)
    for i, (_, row) in enumerate(imp.iterrows()):
        ax.text(row["shap"] + 0.0002, i, f"{row['shap']:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top 15 Predictive Features (SHAP)", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


#  Main 
def main():
    print("\n" + "="*60)
    print("  StockSense AI — Training Pipeline")
    print("="*60)

    print("\n[1/6] Generating synthetic OHLCV data...")
    tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
    frames  = []
    for i, t in enumerate(tickers):
        df_t = generate_ohlcv(n_days=3000, ticker=t, seed=SEED + i)
        df_t = engineer_features(df_t)
        frames.append(df_t)
        print(f"  {t}: {len(df_t)} rows | signal rate: {df_t['target'].mean():.1%}")
    df_all = pd.concat(frames)
    print(f"  Total: {len(df_all):,} rows across {len(tickers)} tickers")

    print("\n[2/6] Building feature matrix...")
    X, y = build_dataset(df_all)
    print(f"  X: {X.shape} | BUY signal rate: {y.mean():.1%}")

    # Temporal split — NO shuffle to prevent lookahead bias
    split_idx = int(len(X) * 0.75)
    X_tr, X_tmp = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_tmp = y.iloc[:split_idx], y.iloc[split_idx:]
    val_idx  = int(len(X_tmp) * 0.5)
    X_val, X_te = X_tmp.iloc[:val_idx], X_tmp.iloc[val_idx:]
    y_val, y_te = y_tmp.iloc[:val_idx], y_tmp.iloc[val_idx:]
    print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}")

    print("\n[3/6] Scaling features...")
    scaler  = StandardScaler()
    X_tr_s  = pd.DataFrame(scaler.fit_transform(X_tr),  columns=X_tr.columns)
    X_val_s = pd.DataFrame(scaler.transform(X_val),     columns=X_val.columns)
    X_te_s  = pd.DataFrame(scaler.transform(X_te),      columns=X_te.columns)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")

    print("\n[4/6] Training XGBoost...")
    xgb_params = dict(
        max_depth=5, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.75, reg_alpha=0.15, reg_lambda=1.2,
        min_child_weight=5, gamma=0.2, n_jobs=-1, verbosity=0,
    )
    cv_scores = cross_val_score(
        xgb.XGBClassifier(n_estimators=400, random_state=SEED, eval_metric="auc", **xgb_params),
        X_tr_s, y_tr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc", n_jobs=-1,
    )
    print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model = xgb.XGBClassifier(
        n_estimators=600, random_state=SEED, eval_metric="auc",
        early_stopping_rounds=40, **xgb_params,
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    print(f"  Best iteration: {model.best_iteration} | Val AUC: {model.best_score:.4f}")
    joblib.dump(model, ARTIFACTS_DIR / "model.pkl")

    print("\n[5/6] Evaluating on test set...")
    y_prob = model.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_te, y_prob)
    f1     = f1_score(y_te, y_pred)
    acc    = accuracy_score(y_te, y_pred)
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=["DOWN/HOLD","UP/BUY"]))
    plot_evaluation(y_te, y_pred, y_prob, ARTIFACTS_DIR)

    print("\n[6/6] SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    sv        = explainer(X_te_s)
    sv_vals   = sv.values
    exp_val   = float(np.ravel(explainer.expected_value)[0])
    joblib.dump(explainer, ARTIFACTS_DIR / "shap_explainer.pkl")
    plot_shap_importance(sv_vals, list(X_te_s.columns), ARTIFACTS_DIR)

    metadata = {
        "model_version":   "1.0.0",
        "trained_at":      pd.Timestamp.now().isoformat(),
        "features":        list(X_tr.columns),
        "feature_display": FEATURE_DISPLAY,
        "threshold":       0.50,
        "forward_days":    FORWARD_DAYS,
        "target_threshold_pct": THRESHOLD_PCT,
        "tickers_trained": tickers,
        "test_auc":        round(auc, 4),
        "test_f1":         round(f1, 4),
        "test_accuracy":   round(acc, 4),
        "baseline_value":  exp_val,
        "n_train":         int(len(X_tr)),
        "n_test":          int(len(X_te)),
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n  Artifacts:")
    for p in sorted(ARTIFACTS_DIR.iterdir()):
        print(f"    {p.name:<35} {p.stat().st_size/1024:>7.1f} KB")




if __name__ == "__main__":
    main()
