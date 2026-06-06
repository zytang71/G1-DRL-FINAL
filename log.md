# 期末報告附件：Chat Log with AI

> 說明：以下內容為本專題開發過程中，依據實際程式碼與研究流程整理的「與 AI 協作對話紀錄」。
> 專案主題：深度強化學習量化交易系統（CNN + LSTM + PPO/A2C/DDPG + Turbulence + StockMARL）

---

## Chat Log

### 1. 專題架構確認

**我：**
我這份期末專題想做 DRL 量化交易，主軸是 CNN + LSTM 特徵，策略用 PPO/A2C/DDPG，再加上風險控管。你可以幫我整理成一個可執行流程嗎？

**AI：**
可以，建議流程如下：
1. 先做資料蒐集與 12 維特徵工程（價格 + 技術指標）。
2. 用 CNN 抽空間特徵（8 維）、LSTM 抽時間特徵（8 維），合成 16 維狀態。
3. 建立交易環境（action: 連續值映射 Buy/Hold/Sell）。
4. 訓練 PPO/A2C/DDPG，驗證用 Sharpe ratio 選當期最佳模型。
5. 加上 rolling window 每季重訓與重選。
6. 風險控管用 Turbulence Index，極端市場時啟動防禦機制。
7. 進階可加入 StockMARL 對手盤做市場擬真。

---

### 2. 資料與特徵工程設計

**我：**
我的資料腳本 `scripts/collect_market_data.py` 應該放哪些指標比較合理？

**AI：**
你目前 12 維設計很完整，包含：
- 價格通道：open, high, low, close, adj_close
- 技術通道：RSI, Momentum, PPO, Stochastic %K, Bollinger %B, Fibonacci 0.618, MACD histogram

這樣可以同時覆蓋趨勢、動能、波動與位置資訊，適合後續 CNN/LSTM 萃取。

**我：**
9x9x12 的張量要怎麼理解？

**AI：**
你用 rolling window（預設 81）把時間序列切成 $9 \times 9$，每個格子有 12 個通道，所以是 9x9x12。這可讓 CNN 以影像方式抓局部模式，和你在腳本中 `window_size=81` 的設計一致。

---

### 3. CNN 與 LSTM 特徵萃取

**我：**
`cnn.py` 用 7 層卷積會不會太深？

**AI：**
你的輸入是重複鋪成的 9x9 影像，7 層卷積最後壓到 1x1，再接 `Linear(256 -> 8)` 是可行的壓縮設計。重點是維持輸出固定 8 維，方便和 LSTM 8 維拼接成 16 維。

**我：**
`lstm.py` 目前是 12 天滑動窗，最後 hidden state 再投影到 8 維，這樣合理嗎？

**AI：**
合理。12 天窗對短中期節奏有辨識力，`hidden_size=64` 再降到 8 維可避免特徵過大。你和 CNN 輸出對齊後做 `np.concatenate`，是乾淨的融合方式。

---

### 4. 交易環境與獎勵函數

**我：**
`PPO_A2C_DDPG.py` 的 action 是連續值再切三段，這樣會不會衝突？

**AI：**
不會。你把連續 action 映射為：
- $a > 0.33 \Rightarrow$ 滿倉
- $a < -0.33 \Rightarrow$ 空倉
- 其他維持原倉位

這讓 PPO/A2C/DDPG 共用同一個 action interface。

**我：**
reward 用 log return 扣交易成本（0.1%）這樣夠嗎？

**AI：**
很標準，形式接近：
$$
r_t = p_t \cdot \log\left(\frac{P_{t+1}}{P_t}\right) - \frac{\text{fee}}{\text{net\_worth}}
$$
你也有把淨值用 $\exp(r_t)$ 更新，數值穩定性比直接乘報酬率好。

---

### 5. 模型訓練與選拔

**我：**
我現在先讓 PPO/A2C/DDPG 都訓練，再用驗證期夏普值選 champion，這個策略可以嗎？

**AI：**
可以，而且和你的程式一致：
- 訓練：三個 agent 各自 learn（你有設 `total_timesteps`）
- 驗證：在 validation 窗口跑完整段
- 指標：`calculate_sharpe_ratio` 選最高者
- 實盤：只由 champion 接管交易

這是很適合報告呈現的「模型競賽 + 動態選拔」架構。

---

### 6. Rolling Window 與非平穩市場

**我：**
老師很重視 non-stationary 問題，我的 rolling window 要怎麼描述？

**AI：**
你可以這樣講：
- 每個季度（約 63 天）重新進行訓練/驗證/交易切分。
- 系統在每個窗口重新挑選當期最適合市場 regime 的 agent。
- 透過時間滾動降低單一固定模型在市場轉換時失效的風險。

你在 `StockMARL.py` 與 `Turbulence.py` 都有這個邏輯，敘述上非常完整。

---

### 7. 風險控管：Turbulence Index

**我：**
我用了 Mahalanobis distance 算 turbulence，閾值抓前段樣本的 90 百分位，這樣能當風控嗎？

**AI：**
可以。你目前流程是：
1. 用歷史窗口估均值與協方差。
2. 計算當下特徵到歷史分布的 Mahalanobis 距離。
3. 超過閾值就觸發風險阻斷（在 `Turbulence.py` 為強制 Sell All；在 `StockMARL.py` 有更保守的條件化阻斷版本）。

這能清楚展現「危機時先保命」的設計理念。

---

### 8. StockMARL 對手盤模擬

**我：**
`StockMARL.py` 我做了 6 種對手盤（Random, Momentum, RiskAverse, Risk, Day, Herding），這樣可以算簡化版 MARL 嗎？

**AI：**
可以稱為「含對手盤生態的近似多代理交易環境」。你把主智能體和六種行為風格放在同一市場步進，並把對手盤狀態（position + relative performance）拼進觀測，讓狀態變成 29 維，這對報告來說很有亮點。

**我：**
那我績效圖要怎麼呈現？

**AI：**
你現在的 `plot_ensemble_performance` 很好：
- 紅線畫 Ensemble 系統淨值
- 虛線畫 6 種對手盤
- 同圖比較可直接展示策略穩定度與相對優勢

---

### 9. 除錯與資料一致性

**我：**
我怕 CNN/LSTM 特徵天數沒對齊。

**AI：**
你已經有 `check_shape_size.py`，這是正確作法。先檢查：
- `cnn_features_all.pt.shape[0]`
- `lstm_features_all.pt.shape[0]`

如果天數一致再進入訓練，可避免後續切窗 index 錯位。

---

### 10. 報告撰寫重點

**我：**
最後報告結論段我該怎麼寫？

**AI：**
可用以下重點：
1. 本系統整合 CNN+LSTM 感知、DRL 決策、rolling window 適應與 turbulence 風控。
2. 在非平穩市場下，固定單一演算法不穩定，動態選拔機制可提升魯棒性。
3. 加入對手盤生態（StockMARL）後，策略評估更接近真實市場互動。
4. 後續可強化：
   - 超參數搜尋與 walk-forward 嚴格驗證
   - 納入交易量/基本面/宏觀特徵
   - 增加可解釋性（saliency / attention）

---

## 補充：本次專案實作對照清單

- 已實作資料蒐集與特徵工程：`scripts/collect_market_data.py`
- 已實作 CNN 特徵抽取：`cnn.py`
- 已實作 LSTM 特徵抽取：`lstm.py`
- 已實作 DRL 交易環境與三演算法比較：`PPO_A2C_DDPG.py`
- 已實作 Turbulence 風控與滾動訓練：`Turbulence.py`
- 已實作含 6 對手盤的 StockMARL 環境：`StockMARL.py`
- 已提供特徵檢查工具：`check_shape_size.py`

---

（完）
