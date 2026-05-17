import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from sklearn.preprocessing import StandardScaler

# 配置系統環境變數，優化 OpenMP 執行緒管理，確保多重運行庫環境下的運算穩定性
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =================================================================
# 1. 量化交易環境定義 (Trading Environment)
# =================================================================
class TradingEnv(gym.Env):
    """
    基於強化學習架構的自定義交易環境。
    整合 16 維時空特徵與 1 維動態持倉資訊，構建 17 維狀態空間。
    """
    def __init__(self, features_16d, prices, turbulence_indices=None, turbulence_threshold=np.inf, initial_balance=1000000, mode='train'):
        super(TradingEnv, self).__init__()
        self.features_16d = features_16d  # 預提取的 CNN+LSTM 融合特徵
        self.prices = prices              # 對應時間序列的市場價格
        self.turbulence_indices = turbulence_indices if turbulence_indices is not None else np.zeros(len(prices))
        self.turbulence_threshold = turbulence_threshold
        self.initial_balance = initial_balance
        self.num_days = len(features_16d)
        self.mode = mode                  # 運作模式：分為訓練 (train) 與測試 (test)
        
        # 動作空間定義：採用 [-1, 1] 的連續區間，以實現不同強化學習演算法的統一接口
        # 邏輯映射：高於 0.33 定義為買入，低於 -0.33 定義為賣出，其餘為持倉不動
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 狀態空間定義：16 維市場特徵加上 1 維智能體當前持倉狀態 (0 或 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        初始化環境狀態，並根據模式決定起始時間點。
        """
        super().reset(seed=seed)
        
        # 隨機化起點策略：在訓練模式下隨機選取起始點，提升模型對於不同市場週期的泛化能力
        if self.mode == 'train':
            self.current_step = np.random.randint(0, self.num_days - 150)
        else:
            self.current_step = 0
            
        self.position = 0.0      # 當前持倉狀態 (0.0 代表空倉，1.0 代表滿倉)
        self.net_worth = self.initial_balance
        self.history_net_worth = [self.initial_balance]
        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        合成 17 維狀態向量。
        將靜態特徵庫中的 16 維市場資訊與智能體的動態持倉狀態進行拼接。
        """
        f_16 = self.features_16d[self.current_step]
        observation = np.append(f_16, self.position)
        return observation.astype(np.float32)

    def step(self, action_input):
        """
        執行決策邏輯，計算獎勵函數並更新淨值。
        """
        # 將連續動作信號映射至交易指令
        action_signal = action_input[0]
        if action_signal > 0.33:
            target_position = 1.0    # 執行买入動作
        elif action_signal < -0.33:
            target_position = 0.0    # 執行平倉動作
        else:
            target_position = self.position # 維持當前持倉
            
        # 極端風險阻斷：亂流指數 (Turbulence Index) 檢查
        current_turbulence = self.turbulence_indices[self.current_step]
        if current_turbulence > self.turbulence_threshold:
            # 觸發紅色警報：強制阻斷買入，啟動一鍵全面清倉 (Sell All)
            target_position = 0.0
            
        current_price = self.prices[self.current_step]
        next_price = self.prices[min(self.current_step + 1, self.num_days - 1)]
        
        # 交易成本建模：基於週轉率計算 0.1% 的手續費與滑價損失
        trade_volume = abs(target_position - self.position)
        transaction_fee = trade_volume * self.net_worth * 0.001
        
        # 獎勵函數設計：採用對數收益 (Log Return) 提升數值穩定性
        # 計算公式：持有部位 × 價格變動對數值 - 交易成本占比
        price_return_log = np.log(next_price / current_price)
        reward = (target_position * price_return_log) - (transaction_fee / self.net_worth)
        
        # 更新帳戶淨值與持倉狀態
        self.net_worth *= np.exp(reward)
        self.position = target_position
        self.history_net_worth.append(self.net_worth)
        
        self.current_step += 1
        
        # 檢查是否到達數據序列終點
        done = self.current_step >= self.num_days - 1
        return self._get_observation(), reward, done, False, {}

# =================================================================
# 2. 性能評估工具 (Metrics)
# =================================================================
def calculate_sharpe_ratio(net_worth_history):
    """
    計算夏普值，評估單位風險下的超額回報。
    採用年化係數 252 個交易日進行標準化。
    """
    returns = pd.Series(net_worth_history).pct_change().dropna()
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def calculate_turbulence_indices(features, window=252):
    """
    計算全時段的亂流指數 (Turbulence Index)。
    使用滾動窗口 (預設 252 天) 的歷史數據計算協方差矩陣與平均值，再計算馬哈拉諾比斯距離。
    """
    n_samples, n_features = features.shape
    turbulence_indices = np.zeros(n_samples)
    
    for i in range(n_samples):
        if i < 2:
            turbulence_indices[i] = 0.0
            continue
            
        # 考慮到初期樣本數過少導致協方差矩陣奇異(Singular)，可設定最小歷史樣本數或加上微小對角線值
        start_idx = max(0, i - window)
        historical_data = features[start_idx:i]
        
        # 樣本太少無法計算穩定的協方差矩陣
        if len(historical_data) < 10:
            turbulence_indices[i] = 0.0
            continue
            
        mu = np.mean(historical_data, axis=0)
        cov_matrix = np.cov(historical_data, rowvar=False)
        
        # 加入微小對角項避免奇異矩陣
        cov_matrix += np.eye(n_features) * 1e-6
        
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            diff = features[i] - mu
            # Mahalanobis distance squared
            mahala_dist = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
            turbulence_indices[i] = mahala_dist
        except np.linalg.LinAlgError:
            turbulence_indices[i] = 0.0
            
    return turbulence_indices

# =================================================================
# 3. 集成交易系統執行邏輯 (Ensemble Workflow - 比較模式)
# =================================================================
def run_quantitative_ensemble_system():
    # A. 數據加載與特徵工程 (保持不變)
    cnn_features = torch.load('0050/cnn_features_all.pt', map_location='cpu', weights_only=True).numpy()
    lstm_features = torch.load('0050/lstm_features_all.pt', map_location='cpu', weights_only=True).numpy()
    features_16d = np.concatenate([cnn_features, lstm_features], axis=1)
    
    scaler = StandardScaler()
    features_16d = scaler.fit_transform(features_16d)
    
    market_df = pd.read_csv('0050/0050_tw.csv')
    market_prices = market_df['close'].values
    
    # 計算全時段的亂流指數
    print("正在計算亂流指數 (Turbulence Index)...")
    turbulence_indices = calculate_turbulence_indices(features_16d, window=252)
    # 取前 2000 天 (訓練初期) 的 90 分位數作為亂流閾值
    threshold_calc_period = turbulence_indices[:2000]
    turbulence_threshold = np.percentile(threshold_calc_period[threshold_calc_period > 0], 90)
    print(f"計算完成。設定亂流閾值 (90th percentile) 為: {turbulence_threshold:.4f}")
    
    # B. 建立滾動視窗訓練機制 (Rolling Window)
    # 初始訓練邊界
    start_time = 2000
    window_size = 63  # 約為 3 個月 (一季度) 的交易日
    
    compute_device = "cpu"
    print(f"系統初始化完成，正在使用裝置: {compute_device}")
    
    # 用於儲存整段實戰交易的歷史淨值與結果
    total_trade_history = []
    current_balance = 1000000
    
    results_summary = []
    
    current_t = start_time
    total_days = len(market_prices)
    
    print("\n" + "="*60)
    print("開始動態適應：滾動時間軸訓練與實盤交易")
    print("="*60)
    
    while current_t + window_size <= total_days:
        train_end = current_t - window_size
        val_start = current_t - window_size
        val_end = current_t
        trade_start = current_t
        trade_end = min(current_t + window_size, total_days)
        
        print(f"\n--- [滾動窗口] 交易區間: {trade_start} ~ {trade_end} ---")
        
        # 建立各階段環境
        env_train = TradingEnv(
            features_16d[:train_end], 
            market_prices[:train_end], 
            turbulence_indices[:train_end],
            turbulence_threshold,
            mode='train'
        )
        env_val = TradingEnv(
            features_16d[val_start:val_end], 
            market_prices[val_start:val_end], 
            turbulence_indices[val_start:val_end],
            turbulence_threshold,
            mode='test'
        )
        env_trade = TradingEnv(
            features_16d[trade_start:trade_end], 
            market_prices[trade_start:trade_end], 
            turbulence_indices[trade_start:trade_end],
            turbulence_threshold,
            initial_balance=current_balance,  # 繼承上一季度的資金
            mode='test'
        )
        
        # 階段 1：訓練 (Training) - 使用過去歷史數據同步進行獨立訓練
        print(">> 階段 1：訓練 (Training) - PPO, A2C, DDPG...")
        agent_ppo = PPO("MlpPolicy", env_train, device=compute_device, verbose=0).learn(total_timesteps=5000)
        agent_a2c = A2C("MlpPolicy", env_train, device=compute_device, verbose=0).learn(total_timesteps=5000)
        
        action_dimension = env_train.action_space.shape[-1]
        exploration_noise = NormalActionNoise(mean=np.zeros(action_dimension), sigma=0.1 * np.ones(action_dimension))
        agent_ddpg = DDPG("MlpPolicy", env_train, action_noise=exploration_noise, device=compute_device, verbose=0).learn(total_timesteps=5000)
        
        ensemble_pool = [agent_ppo, agent_a2c, agent_ddpg]
        agent_names = ["PPO", "A2C", "DDPG"]
        
        # 階段 2：驗證 (Validation) - 3個月的滾動窗口，以夏普值為唯一標準
        print(">> 階段 2：驗證 (Validation) - 評估夏普值...")
        best_sharpe = -np.inf
        best_agent = None
        best_agent_name = ""
        
        for agent, name in zip(ensemble_pool, agent_names):
            obs, _ = env_val.reset()
            is_done = False
            while not is_done:
                action, _ = agent.predict(obs)
                obs, _, is_done, _, _ = env_val.step(action)
            
            sharpe_val = calculate_sharpe_ratio(env_val.history_net_worth)
            print(f"   - {name} 驗證期夏普值: {sharpe_val:.4f}")
            
            if sharpe_val > best_sharpe:
                best_sharpe = sharpe_val
                best_agent = agent
                best_agent_name = name
                
        print(f"   => 優勝 Agent: {best_agent_name} (Sharpe: {best_sharpe:.4f})")
        
        # 階段 3：實盤交易 (Trading) - 由勝出的 Agent 全權執行決策
        print(f">> 階段 3：實盤交易 (Trading) - 使用 {best_agent_name} 進行交易...")
        obs, _ = env_trade.reset()
        is_done = False
        
        turbulence_triggers = 0
        while not is_done:
            if env_trade.turbulence_indices[env_trade.current_step] > env_trade.turbulence_threshold:
                turbulence_triggers += 1
            action, _ = best_agent.predict(obs)
            obs, _, is_done, _, _ = env_trade.step(action)
            
        print(f"   - 本季觸發亂流阻斷次數: {turbulence_triggers}")
            
        current_balance = env_trade.history_net_worth[-1]
        
        if len(total_trade_history) == 0:
            total_trade_history.extend(env_trade.history_net_worth)
        else:
            total_trade_history.extend(env_trade.history_net_worth[1:])
            
        results_summary.append({
            "Trade_Start": trade_start,
            "Trade_End": trade_end,
            "Winning_Agent": best_agent_name,
            "Validation_Sharpe": best_sharpe,
            "Quarter_Final_Balance": current_balance,
            "Turbulence_Triggers": turbulence_triggers
        })
        
        # 推進時間軸
        current_t += window_size

    # F. 綜合結算對比報告
    initial_cap = 1000000
    final_equity = current_balance
    roi = (final_equity - initial_cap) / initial_cap * 100
    trading_sharpe = calculate_sharpe_ratio(total_trade_history)
    
    print(f"\n" + "="*60)
    print("全期實盤滾動交易總結報告")
    print("="*60)
    print(f"{'季度 (Days)':<15} | {'優勝演算法':<10} | {'季末淨值':<15} | {'亂流觸發次數':<12}")
    print("-" * 60)
    for res in results_summary:
        quarter_str = f"{res['Trade_Start']}~{res['Trade_End']}"
        print(f"{quarter_str:<15} | {res['Winning_Agent']:<10} | {res['Quarter_Final_Balance']:>15,.2f} | {res['Turbulence_Triggers']:>12}")
        
    print("-" * 60)
    print(f"初始資金: {initial_cap:,.2f}")
    print(f"最終資金: {final_equity:,.2f}")
    print(f"累積報酬率: {roi:.2f}%")
    print(f"全期夏普值: {trading_sharpe:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_quantitative_ensemble_system()