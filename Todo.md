# TODO

依 proposal 的簡化執行清單：

- [ ] 整理資料集與欄位定義（股價、技術指標、宏觀指標）。
- [ ] 建立 `9x9x12` 特徵張量輸入格式。
- [ ] 完成 `CNN + LSTM` 感知模型與 `Buy/Sell/Hold` 決策介面。
- [ ] 訓練基礎 DRL 策略（PPO / A2C / DDPG）。
- [ ] 實作 rolling window 訓練與驗證流程（3 個月輪替）。
- [ ] 加入 `Turbulence Index` 風險阻斷機制（必要時 Sell All）。
- [ ] 建立簡化版 StockMARL 對手盤環境。
- [ ] 完成回測報告（Return、Sharpe、Volatility、Max Drawdown）。
- [ ] 產出模型可解釋性圖表（Saliency / Attention）。
