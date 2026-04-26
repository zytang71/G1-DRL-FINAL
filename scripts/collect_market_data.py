#!/usr/bin/env python3
"""Collect market data with yfinance and build 12-channel feature datasets.

The output includes:
1) Raw price CSV per symbol.
2) Feature CSV per symbol with 12 channels:
   - 5 price channels: open, high, low, close, adj_close
   - 7 indicators: RSI, Momentum, PPO, Stochastic %K, Bollinger %B, Fibonacci, MACD histogram
3) Optional 9x9x12 rolling tensors saved as .npz with channels-first layout (N, 12, 9, 9).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_SYMBOLS = [
    "0050.TW",  # Yuanta Taiwan Top 50 ETF
    "2330.TW",  # TSMC (TW)
    "^GSPC",  # S&P 500 index
    "^DJI",  # Dow Jones Industrial Average
    "^IXIC",  # Nasdaq Composite
    "SPY",  # S&P 500 ETF
    "QQQ",  # Nasdaq-100 ETF
    "TSM",  # TSMC ADR (US)
    "AAPL",  # Apple
]

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "rsi_14",
    "momentum_10",
    "ppo_12_26",
    "stoch_k_14",
    "bb_percent_b_20",
    "fibonacci_0_618_20",
    "macd_hist_12_26_9",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect market data and build feature datasets.")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat(), help="End date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1d", help="yfinance interval (default: 1d).")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Tickers list, e.g. --symbols 0050.TW 2330.TW ^GSPC",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Optional JSON file. Supported formats: ['AAPL','MSFT'] or {'symbols':['AAPL','MSFT']}.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/market"), help="Output directory root.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional yfinance cache base directory. Default: <output-root>/.yfinance_cache",
    )
    parser.add_argument("--window-size", type=int, default=81, help="Rolling window length for 9x9 tensors.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for rolling window.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not min-max normalize each channel in each rolling window before tensor export.",
    )
    parser.add_argument(
        "--skip-tensor",
        action="store_true",
        help="Skip tensor generation and only export raw/features CSV.",
    )
    return parser.parse_args()


def sanitize_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_").lower()


def parse_symbols(raw_symbols: Sequence[str]) -> list[str]:
    symbols: list[str] = []
    for item in raw_symbols:
        symbols.extend(s.strip() for s in item.split(",") if s.strip())
    return symbols


def load_symbols(symbols_arg: Sequence[str], symbols_file: Path | None) -> list[str]:
    symbols = parse_symbols(symbols_arg)
    if symbols_file is None:
        return symbols

    payload = json.loads(symbols_file.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        file_symbols = [str(x).strip() for x in payload if str(x).strip()]
    elif isinstance(payload, dict) and isinstance(payload.get("symbols"), list):
        file_symbols = [str(x).strip() for x in payload["symbols"] if str(x).strip()]
    else:
        raise ValueError("symbols-file format invalid; use list or {'symbols': [...]} JSON.")

    return file_symbols or symbols


def fetch_symbol_history(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return (field, ticker) columns.
        df.columns = [str(c[0]) for c in df.columns]

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ppo(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ((ema_fast - ema_slow) / ema_slow) * 100


def compute_macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Open", "High", "Low", "Close", "Adj Close"}
    missing = required_cols.difference(raw.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    high = raw["High"].astype(float)
    low = raw["Low"].astype(float)
    close = raw["Close"].astype(float)
    adj_close = raw["Adj Close"].astype(float)

    out = pd.DataFrame(index=raw.index)
    out["open"] = raw["Open"].astype(float)
    out["high"] = high
    out["low"] = low
    out["close"] = close
    out["adj_close"] = adj_close

    out["rsi_14"] = compute_rsi(adj_close, period=14)
    out["momentum_10"] = adj_close.pct_change(10)
    out["ppo_12_26"] = compute_ppo(adj_close, fast=12, slow=26)

    rolling_high_14 = high.rolling(window=14).max()
    rolling_low_14 = low.rolling(window=14).min()
    denom_14 = (rolling_high_14 - rolling_low_14).replace(0, np.nan)
    out["stoch_k_14"] = ((close - rolling_low_14) / denom_14) * 100

    rolling_mean_20 = adj_close.rolling(window=20).mean()
    rolling_std_20 = adj_close.rolling(window=20).std()
    bb_upper = rolling_mean_20 + (2 * rolling_std_20)
    bb_lower = rolling_mean_20 - (2 * rolling_std_20)
    bb_denom = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_percent_b_20"] = (adj_close - bb_lower) / bb_denom

    fib_high_20 = high.rolling(window=20).max()
    fib_low_20 = low.rolling(window=20).min()
    fib_range = (fib_high_20 - fib_low_20).replace(0, np.nan)
    fib_618 = fib_high_20 - (fib_range * 0.618)
    out["fibonacci_0_618_20"] = (adj_close - fib_618) / fib_range

    out["macd_hist_12_26_9"] = compute_macd_hist(adj_close, fast=12, slow=26, signal=9)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def minmax_normalize_window(window: np.ndarray) -> np.ndarray:
    mins = np.nanmin(window, axis=0, keepdims=True)
    maxs = np.nanmax(window, axis=0, keepdims=True)
    span = maxs - mins
    span[span == 0] = 1.0
    return (window - mins) / span


def build_rolling_tensors(
    features: pd.DataFrame,
    window_size: int,
    stride: int,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be > 0")
    side = int(math.isqrt(window_size))
    if side * side != window_size:
        raise ValueError(f"window_size must be a perfect square (got {window_size})")

    values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if len(values) < window_size:
        return np.empty((0, len(FEATURE_COLUMNS), side, side), dtype=np.float32), np.empty((0,), dtype="datetime64[ns]")

    tensors: list[np.ndarray] = []
    end_dates: list[pd.Timestamp] = []
    for end_idx in range(window_size, len(values) + 1, stride):
        window = values[end_idx - window_size : end_idx]
        if normalize:
            window = minmax_normalize_window(window)
        image_hwc = window.reshape(side, side, len(FEATURE_COLUMNS))
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        tensors.append(image_chw.astype(np.float32))
        end_dates.append(features.index[end_idx - 1])

    tensor_array = np.stack(tensors, axis=0) if tensors else np.empty((0, len(FEATURE_COLUMNS), side, side), dtype=np.float32)
    date_array = np.array(end_dates, dtype="datetime64[ns]")
    return tensor_array, date_array


def configure_yfinance_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # yfinance uses sqlite caches. In restricted environments, default user cache may be unwritable.
    if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
        yf.cache.set_cache_location(str(cache_dir))


def main() -> None:
    args = parse_args()
    symbols = load_symbols(args.symbols, args.symbols_file)
    if not symbols:
        raise ValueError("no symbols provided")

    output_root = args.output_root
    raw_dir = output_root / "raw"
    feature_dir = output_root / "features_12d"
    tensor_dir = output_root / "tensors_9x9x12"
    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_tensor:
        tensor_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir if args.cache_dir is not None else output_root / ".yfinance_cache"
    configure_yfinance_cache(cache_dir)

    summary: dict[str, object] = {
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "yfinance_cache_dir": str(cache_dir.as_posix()),
        "window_size": args.window_size,
        "stride": args.stride,
        "normalize_windows": not args.no_normalize,
        "feature_columns": FEATURE_COLUMNS,
        "symbols": [],
    }

    for symbol in symbols:
        raw = fetch_symbol_history(symbol, start=args.start, end=args.end, interval=args.interval)
        if raw.empty:
            summary["symbols"].append(
                {
                    "symbol": symbol,
                    "status": "empty",
                    "raw_rows": 0,
                    "feature_rows": 0,
                    "tensor_samples": 0,
                }
            )
            continue

        features = build_feature_frame(raw)
        safe_symbol = sanitize_symbol(symbol)
        raw_path = raw_dir / f"{safe_symbol}.csv"
        feature_path = feature_dir / f"{safe_symbol}.csv"

        raw.to_csv(raw_path, index_label="date")
        features.to_csv(feature_path, index_label="date")

        tensor_samples = 0
        tensor_path = None
        if not args.skip_tensor:
            tensors, end_dates = build_rolling_tensors(
                features=features,
                window_size=args.window_size,
                stride=args.stride,
                normalize=not args.no_normalize,
            )
            tensor_samples = int(tensors.shape[0])
            tensor_path = tensor_dir / f"{safe_symbol}.npz"
            np.savez_compressed(
                tensor_path,
                symbol=symbol,
                tensors=tensors,
                dates=end_dates.astype("datetime64[D]").astype(str),
                feature_columns=np.array(FEATURE_COLUMNS),
            )

        summary["symbols"].append(
            {
                "symbol": symbol,
                "status": "ok",
                "raw_rows": int(len(raw)),
                "feature_rows": int(len(features)),
                "tensor_samples": tensor_samples,
                "raw_path": str(raw_path.as_posix()),
                "feature_path": str(feature_path.as_posix()),
                "tensor_path": str(tensor_path.as_posix()) if tensor_path else None,
            }
        )

    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
