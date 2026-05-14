import os
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
    def __init__(self, features_16d, prices, initial_balance=1000000, mode='train'):
        super(TradingEnv, self).__init__()
        self.features_16d = features_16d  # 預提取的 CNN+LSTM 融合特徵
        self.prices = prices              # 對應時間序列的市場價格
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

# =================================================================
# 3. 集成交易系統執行邏輯 (Ensemble Workflow - 比較模式)
# =================================================================
def run_quantitative_ensemble_system():
    # A. 數據加載與特徵工程 (保持不變)
    cnn_features = torch.load('cnn_features_all.pt', map_location='cpu', weights_only=True).numpy()
    lstm_features = torch.load('lstm_features_all.pt', map_location='cpu', weights_only=True).numpy()
    features_16d = np.concatenate([cnn_features, lstm_features], axis=1)
    
    scaler = StandardScaler()
    features_16d = scaler.fit_transform(features_16d)
    
    market_df = pd.read_csv('0050_tw.csv')
    market_prices = market_df['close'].values
    
    # B. 建立滾動視窗與分段環境
    training_boundary = 2000
    validation_boundary = 2300
    
    env_train = TradingEnv(features_16d[:training_boundary], market_prices[:training_boundary], mode='train')
    env_val = TradingEnv(features_16d[training_boundary:validation_boundary], market_prices[training_boundary:validation_boundary], mode='test')
    env_trade = TradingEnv(features_16d[validation_boundary:], market_prices[validation_boundary:], mode='test')
    
    compute_device = "cpu"
    print(f"系統初始化完成，正在使用裝置: {compute_device}")

    # C. 多智能體並行訓練階段
    print("\n[1/3] 啟動 PPO 穩定趨勢智能體訓練...")
    agent_ppo = PPO("MlpPolicy", env_train, device=compute_device, verbose=0).learn(total_timesteps=10000)
    
    print("[2/3] 啟動 A2C 靈敏反應智能體訓練...")
    agent_a2c = A2C("MlpPolicy", env_train, device=compute_device, verbose=0).learn(total_timesteps=10000)
    
    print("[3/3] 啟動 DDPG 精確控盤智能體訓練...")
    action_dimension = env_train.action_space.shape[-1]
    exploration_noise = NormalActionNoise(mean=np.zeros(action_dimension), sigma=0.1 * np.ones(action_dimension))
    agent_ddpg = DDPG("MlpPolicy", env_train, action_noise=exploration_noise, device=compute_device, verbose=0).learn(total_timesteps=10000)

    ensemble_pool = [agent_ppo, agent_a2c, agent_ddpg]
    agent_names = ["PPO", "A2C", "DDPG"]
    
    # D. 驗證階段 (單純記錄表現，不進行篩選)
    print("\n--- 驗證階段表現預覽 (300 天) ---")
    for agent, name in zip(ensemble_pool, agent_names):
        obs, _ = env_val.reset()
        is_done = False
        while not is_done:
            action, _ = agent.predict(obs)
            obs, _, is_done, _, _ = env_val.step(action)
        
        sharpe_val = calculate_sharpe_ratio(env_val.history_net_worth)
        print(f"智能體 {name} 驗證期夏普值: {sharpe_val:.4f}")

    # E. 最終實戰階段對比 (Final Comparative Backtesting)
    print(f"\n--- 最終實戰階段回測 (總計 {len(market_prices[validation_boundary:])} 個交易日) ---")
    
    # 用於儲存各智能體表現結果
    results_summary = []

    for agent, name in zip(ensemble_pool, agent_names):
        obs, _ = env_trade.reset()
        is_done = False
        while not is_done:
            action, _ = agent.predict(obs)
            obs, _, is_done, _, _ = env_trade.step(action)
        
        # 統計該智能體表現
        initial_cap = 1000000
        final_equity = env_trade.history_net_worth[-1]
        roi = (final_equity - initial_cap) / initial_cap * 100
        trading_sharpe = calculate_sharpe_ratio(env_trade.history_net_worth)
        
        results_summary.append({
            "Name": name,
            "Final_Equity": final_equity,
            "ROI": roi,
            "Sharpe": trading_sharpe
        })
        print(f"已完成 {name} 的實戰回測。")

    # F. 綜合結算對比報告
    print(f"\n" + "="*50)
    print(f"{'演算法':<10} | {'最終淨值':<15} | {'累積報酬率':<10} | {'實戰夏普值':<10}")
    print("-" * 50)
    
    for res in results_summary:
        print(f"{res['Name']:<10} | {res['Final_Equity']:>15,.2f} | {res['ROI']:>9.2f}% | {res['Sharpe']:>10.4f}")
    
    # 找出本次表現最好的算法
    best_agent = max(results_summary, key=lambda x: x['ROI'])
    print("-" * 50)
    print(f"本次實戰中，獲利能力最強的算法為: {best_agent['Name']}")
    print("="*50)

if __name__ == "__main__":
    run_quantitative_ensemble_system()