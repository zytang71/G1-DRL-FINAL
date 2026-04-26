# Data Collection Quickstart

This step implements the first TODO item: collect market data and define input columns.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run collection

```bash
python scripts/collect_market_data.py --start 2015-01-01 --end 2026-04-23
```

Default symbols include:
- `0050.TW`
- `2330.TW`
- `^GSPC`
- `^DJI`
- `^IXIC`
- `SPY`
- `QQQ`
- `TSM`
- `AAPL`

## 3) Output structure

The script writes into `data/market/`:
- `raw/*.csv`: raw OHLCV from yfinance
- `features_12d/*.csv`: 12-channel features per date
- `tensors_9x9x12/*.npz`: rolling tensors in `(N, 12, 9, 9)` format
- `dataset_summary.json`: per-symbol row/sample summary

## 4) Feature channels (12)

Price channels (5):
- `open`, `high`, `low`, `close`, `adj_close`

Indicator channels (7):
- `rsi_14`
- `momentum_10`
- `ppo_12_26`
- `stoch_k_14`
- `bb_percent_b_20`
- `fibonacci_0_618_20`
- `macd_hist_12_26_9`

## Useful flags

- `--symbols 0050.TW 2330.TW ^GSPC`
- `--symbols-file path/to/symbols.json`
- `--skip-tensor` (only export raw + features)
- `--window-size 81 --stride 1`
- `--no-normalize` (disable per-window min-max normalization)
- `--cache-dir data/market/.yfinance_cache` (set writable yfinance sqlite cache path)
