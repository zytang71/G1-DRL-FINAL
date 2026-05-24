import os
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
stock_name = "spy"

# =================================================================
# 1. 6 種對手盤定義 (Opponent Traders)
# =================================================================
class BaseTrader:
    def __init__(self, name, initial_balance=1000000):
        self.name = name
        self.initial_balance = initial_balance
        self.net_worth = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.history_net_worth = [initial_balance]

    def reset(self):
        self.net_worth = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.history_net_worth = [self.initial_balance]

    def update_performance(self, target_position, current_price, next_price):
        trade_volume = abs(target_position - self.position)
        transaction_fee = trade_volume * self.net_worth * 0.001
        
        price_return_log = np.log(next_price / current_price) if current_price > 0 else 0
        reward = (target_position * price_return_log) - (transaction_fee / self.net_worth)
        
        self.net_worth *= np.exp(reward)
        self.position = target_position
        self.history_net_worth.append(self.net_worth)
        
        if target_position > 0 and self.position == 0:
            self.entry_price = current_price
        elif target_position == 0:
            self.entry_price = 0.0

    def get_action(self, current_step, prices, other_actions=None):
        return 0.0

class RandomTrader(BaseTrader):
    def get_action(self, current_step, prices, other_actions=None):
        return np.random.choice([0.0, 1.0])

class MomentumTrader(BaseTrader):
    def get_action(self, current_step, prices, other_actions=None):
        if current_step < 20:
            return self.position
        ma5 = np.mean(prices[current_step-5:current_step])
        ma20 = np.mean(prices[current_step-20:current_step])
        return 1.0 if ma5 > ma20 else 0.0

class RiskAverseTrader(BaseTrader):
    def get_action(self, current_step, prices, other_actions=None):
        if current_step < 5:
            return 0.0
        current_price = prices[current_step-1]
        if self.position > 0 and self.entry_price > 0:
            ret = (current_price - self.entry_price) / self.entry_price
            if ret <= -0.02 or ret >= 0.05:
                return 0.0
            return 1.0
        else:
            if current_price > np.mean(prices[current_step-5:current_step]):
                return 1.0
            return 0.0

class RiskTrader(BaseTrader):
    def get_action(self, current_step, prices, other_actions=None):
        if current_step < 10:
            return 0.0
        current_price = prices[current_step-1]
        if self.position > 0 and self.entry_price > 0:
            ret = (current_price - self.entry_price) / self.entry_price
            if ret >= 0.15:
                return 0.0
            return 1.0
        else:
            if current_price < np.mean(prices[current_step-10:current_step]) * 0.98:
                return 1.0
            return 0.0

class DayTrader(BaseTrader):
    def get_action(self, current_step, prices, other_actions=None):
        if current_step < 2:
            return 0.0
        if prices[current_step-1] > prices[current_step-2]:
            return 1.0
        else:
            return 0.0

class HerdingTrader(BaseTrader):
    def get_action(self, current_step, prices, other_actions=None):
        if other_actions is None or len(other_actions) == 0:
            return 0.0
        return 1.0 if np.mean(other_actions) > 0.5 else 0.0

# =================================================================
# 2. 亂流指數計算
# =================================================================
def calculate_turbulence_indices(features, window=252):
    n_samples, n_features = features.shape
    turbulence_indices = np.zeros(n_samples)
    
    for i in range(n_samples):
        if i < 2:
            turbulence_indices[i] = 0.0
            continue
            
        start_idx = max(0, i - window)
        historical_data = features[start_idx:i]
        
        if len(historical_data) < 10:
            turbulence_indices[i] = 0.0
            continue
            
        mu = np.mean(historical_data, axis=0)
        cov_matrix = np.cov(historical_data, rowvar=False)
        cov_matrix += np.eye(n_features) * 1e-6
        
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            diff = features[i] - mu
            mahala_dist = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
            turbulence_indices[i] = mahala_dist
        except np.linalg.LinAlgError:
            turbulence_indices[i] = 0.0
            
    return turbulence_indices

# =================================================================
# 3. StockMARL 交易環境定義 (StockMARLEnv)
# =================================================================
class StockMARLEnv(gym.Env):
    def __init__(self, features_16d, prices, turbulence_indices=None, turbulence_threshold=np.inf, 
                 initial_balance=1000000, initial_position=0.0, mode='train', initial_agent_states=None):
        super(StockMARLEnv, self).__init__()
        self.features_16d = features_16d
        self.prices = prices
        self.turbulence_indices = turbulence_indices if turbulence_indices is not None else np.zeros(len(prices))
        self.turbulence_threshold = turbulence_threshold
        
        self.initial_balance = initial_balance
        self.initial_position = initial_position
        self.num_days = len(features_16d)
        self.mode = mode
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
        
        self.agents = {
            'Random': RandomTrader('Random', initial_balance),
            'Momentum': MomentumTrader('Momentum', initial_balance),
            'RiskAverse': RiskAverseTrader('RiskAverse', initial_balance),
            'Risk': RiskTrader('Risk', initial_balance),
            'Day': DayTrader('Day', initial_balance),
            'Herding': HerdingTrader('Herding', initial_balance)
        }
        
        self.initial_agent_states = initial_agent_states

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.mode == 'train':
            self.current_step = np.random.randint(0, max(1, self.num_days - 150))
        else:
            self.current_step = 0
            
        self.position = self.initial_position
        self.net_worth = self.initial_balance
        self.history_net_worth = [self.initial_balance]
        
        for name, agent in self.agents.items():
            if self.initial_agent_states and name in self.initial_agent_states:
                state = self.initial_agent_states[name]
                agent.net_worth = state['net_worth']
                agent.position = state['position']
                agent.entry_price = state['entry_price']
                agent.history_net_worth = [state['net_worth']]
            else:
                agent.reset()
            
        return self._get_observation(), {}

    def _get_observation(self):
        f_16 = self.features_16d[self.current_step]
        opp_obs = []
        for name in ['Random', 'Momentum', 'RiskAverse', 'Risk', 'Day', 'Herding']:
            agent = self.agents[name]
            rel_perf = (agent.net_worth - self.initial_balance) / self.initial_balance
            opp_obs.extend([agent.position, rel_perf])
            
        observation = np.concatenate([f_16, [self.position], opp_obs])
        return observation.astype(np.float32)

    def get_current_agent_states(self):
        states = {}
        for name, agent in self.agents.items():
            states[name] = {
                'net_worth': agent.net_worth,
                'position': agent.position,
                'entry_price': agent.entry_price
            }
        return states

    def step(self, action):
        action_signal = action[0]
        if action_signal > 0.33:
            target_position = 1.0
        elif action_signal < -0.33:
            target_position = 0.0
        else:
            target_position = self.position
                
        # 極端風險阻斷：放寬亂流判定
        current_turbulence = self.turbulence_indices[self.current_step]
        if current_turbulence > self.turbulence_threshold:
            # 只有在大盤下跌時才將亂流視為危機，強制空手觀望避險
            if self.current_step > 0 and self.prices[self.current_step] < self.prices[self.current_step - 1]:
                target_position = 0.0
            
        current_price = self.prices[self.current_step]
        next_price = self.prices[min(self.current_step + 1, self.num_days - 1)]
        
        # 主智能體計算
        trade_volume = abs(target_position - self.position)
        transaction_fee = trade_volume * self.net_worth * 0.001
        price_return_log = np.log(next_price / current_price) if current_price > 0 else 0
        reward = (target_position * price_return_log) - (transaction_fee / self.net_worth)
        
        self.net_worth *= np.exp(reward)
        self.position = target_position
        self.history_net_worth.append(self.net_worth)
        
        # 取得對手盤動作 (對手盤不受亂流影響，呈現未受保護的市場狀態)
        opp_actions = []
        for name in ['Random', 'Momentum', 'RiskAverse', 'Risk', 'Day']:
            act = self.agents[name].get_action(self.current_step, self.prices)
            opp_actions.append(act)
        
        herding_act = self.agents['Herding'].get_action(self.current_step, self.prices, opp_actions)
        opp_actions.append(herding_act)
        
        # 更新對手盤狀態
        for idx, name in enumerate(['Random', 'Momentum', 'RiskAverse', 'Risk', 'Day', 'Herding']):
            self.agents[name].update_performance(opp_actions[idx], current_price, next_price)
            
        self.current_step += 1
        done = self.current_step >= self.num_days - 1
        
        return self._get_observation(), reward, done, False, {}

# =================================================================
# 4. 性能評估工具與視覺化
# =================================================================
def calculate_sharpe_ratio(net_worth_history):
    returns = pd.Series(net_worth_history).pct_change().dropna()
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def plot_ensemble_performance(total_trade_history, opp_trade_histories, title="StockMARL Ensemble vs Opponents: "+f'{stock_name}'):
    plt.figure(figsize=(14, 8))
    
    # 繪製 Ensemble 系統績效
    plt.plot(total_trade_history, label='Ensemble System (適應型集成智能體)', color='red', linewidth=3)
    
    # 繪製 6 種對手盤績效
    colors_opp = ['gray', 'blue', 'green', 'orange', 'purple', 'cyan']
    names = ['Random', 'Momentum', 'RiskAverse', 'Risk', 'Day', 'Herding']
    labels = ['隨機客', '動能交易者', '風險厭惡者', '風險狂熱者', '當沖客', '羊群跟風者']
    
    for idx, name in enumerate(names):
        plt.plot(opp_trade_histories[name], label=f'{name} ({labels[idx]})', color=colors_opp[idx], alpha=0.6, linestyle='--')
        
    plt.title(title)
    plt.xlabel('Trading Days (Rolling Window Horizon)')
    plt.ylabel('Net Worth (Portfolio Value)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'features_and_results/{stock_name}/{stock_name}_performance.png', dpi=300)
    print("已儲存綜合績效走勢圖為" + f'features_and_results/{stock_name}/{stock_name}_performance.png')

# =================================================================
# 5. 集成系統執行邏輯 (Ensemble Workflow)
# =================================================================
def run_quantitative_ensemble_system():
    print("加載特徵與市場資料 ...")
    cnn_features = torch.load(f'features_and_results/{stock_name}/cnn_features_all.pt', map_location='cpu', weights_only=True).numpy()
    lstm_features = torch.load(f'features_and_results/{stock_name}/lstm_features_all.pt', map_location='cpu', weights_only=True).numpy()
    features_16d = np.concatenate([cnn_features, lstm_features], axis=1)
    
    scaler = StandardScaler()
    features_16d = scaler.fit_transform(features_16d)
    
    market_df = pd.read_csv(f'features_and_results/{stock_name}/{stock_name}.csv')
    market_prices = market_df['close'].values
    
    print("正在計算亂流指數 (Turbulence Index)...")
    turbulence_indices = calculate_turbulence_indices(features_16d, window=252)
    threshold_calc_period = turbulence_indices[:2000]
    turbulence_threshold = np.percentile(threshold_calc_period[threshold_calc_period > 0], 90)
    print(f"計算完成。設定亂流閾值 (90th percentile) 為: {turbulence_threshold:.4f}")
    
    # 建立滾動視窗訓練機制 (Rolling Window)
    start_time = 2000
    window_size = 63  # 約為 3 個月 (一季度)
    compute_device = "cpu"
    
    # 全期資金紀錄
    total_trade_history = []
    opp_trade_histories = {name: [] for name in ['Random', 'Momentum', 'RiskAverse', 'Risk', 'Day', 'Herding']}
    
    current_balance = 1000000
    current_position = 0.0
    current_agent_states = None
    results_summary = []
    
    current_t = start_time
    total_days = len(market_prices)
    
    print("\n" + "="*60)
    print("開始動態適應：滾動時間軸訓練與實盤交易 (包含 6 對手盤)")
    print("="*60)
    
    while current_t + window_size <= total_days:
        train_end = current_t - window_size
        val_start = current_t - window_size
        val_end = current_t
        trade_start = current_t
        trade_end = min(current_t + window_size, total_days)
        
        print(f"\n--- [滾動窗口] 交易區間: {trade_start} ~ {trade_end} ---")
        
        # 階段 1：訓練環境準備
        env_train_cont = StockMARLEnv(
            features_16d[:train_end], market_prices[:train_end],
            turbulence_indices[:train_end], turbulence_threshold,
            mode='train'
        )
        
        print(">> 階段 1：訓練 (PPO, A2C, DDPG)...")
        agent_ppo = PPO("MlpPolicy", env_train_cont, device=compute_device, verbose=0).learn(total_timesteps=3000)
        agent_a2c = A2C("MlpPolicy", env_train_cont, device=compute_device, verbose=0).learn(total_timesteps=3000)
        
        action_dim = env_train_cont.action_space.shape[-1]
        noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))
        agent_ddpg = DDPG("MlpPolicy", env_train_cont, action_noise=noise, device=compute_device, verbose=0).learn(total_timesteps=3000)
        
        # 階段 2：驗證選拔
        env_val_cont = StockMARLEnv(
            features_16d[val_start:val_end], market_prices[val_start:val_end],
            turbulence_indices[val_start:val_end], turbulence_threshold,
            mode='test'
        )
        
        print(">> 階段 2：驗證 (評估夏普值選拔 Champion)...")
        best_sharpe = -np.inf
        best_agent = None
        best_agent_name = ""
        
        ensemble_pool = [
            (agent_ppo, "PPO"),
            (agent_a2c, "A2C"),
            (agent_ddpg, "DDPG")
        ]
        
        for agent, name in ensemble_pool:
            env = env_val_cont
            obs, _ = env.reset()
            is_done = False
            while not is_done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, is_done, _, _ = env.step(action)
                
            sharpe_val = calculate_sharpe_ratio(env.history_net_worth)
            if sharpe_val > best_sharpe:
                best_sharpe = sharpe_val
                best_agent = agent
                best_agent_name = name
                
        print(f"   => 優勝模型: {best_agent_name} (Sharpe: {best_sharpe:.4f})")
        
        # 階段 3：實盤交易 (使用上季的資產狀態)
        print(f">> 階段 3：實盤交易 (接管者: {best_agent_name})")
        env_trade = StockMARLEnv(
            features_16d[trade_start:trade_end], market_prices[trade_start:trade_end],
            turbulence_indices[trade_start:trade_end], turbulence_threshold,
            initial_balance=current_balance, initial_position=current_position,
            mode='test', initial_agent_states=current_agent_states
        )
        
        obs, _ = env_trade.reset()
        is_done = False
        turbulence_triggers = 0
        while not is_done:
            if env_trade.turbulence_indices[env_trade.current_step] > env_trade.turbulence_threshold:
                turbulence_triggers += 1
            action, _ = best_agent.predict(obs, deterministic=True)
            obs, _, is_done, _, _ = env_trade.step(action)
            
        print(f"   - 本季觸發亂流阻斷次數: {turbulence_triggers}")
        
        # 儲存與推進狀態
        current_balance = env_trade.history_net_worth[-1]
        current_position = env_trade.position
        current_agent_states = env_trade.get_current_agent_states()
        
        if len(total_trade_history) == 0:
            total_trade_history.extend(env_trade.history_net_worth)
            for name in opp_trade_histories:
                opp_trade_histories[name].extend(env_trade.agents[name].history_net_worth)
        else:
            total_trade_history.extend(env_trade.history_net_worth[1:])
            for name in opp_trade_histories:
                opp_trade_histories[name].extend(env_trade.agents[name].history_net_worth[1:])
                
        results_summary.append({
            "Trade_Start": trade_start,
            "Trade_End": trade_end,
            "Winning_Agent": best_agent_name,
            "Validation_Sharpe": best_sharpe,
            "Quarter_Final_Balance": current_balance,
            "Turbulence_Triggers": turbulence_triggers
        })
        
        current_t += window_size

    # F. 綜合結算對比報告
    initial_cap = 1000000
    final_equity = current_balance
    roi = (final_equity - initial_cap) / initial_cap * 100
    trading_sharpe = calculate_sharpe_ratio(total_trade_history)
    
    print(f"\n" + "="*60)
    print("全期實盤滾動交易總結報告 (Ensemble vs 6 Opponents)")
    print("="*60)
    print(f"{'季度 (Days)':<15} | {'優勝演算法':<10} | {'季末淨值':<15} | {'亂流觸發':<10}")
    print("-" * 60)
    for res in results_summary:
        quarter_str = f"{res['Trade_Start']}~{res['Trade_End']}"
        print(f"{quarter_str:<15} | {res['Winning_Agent']:<10} | {res['Quarter_Final_Balance']:>15,.2f} | {res['Turbulence_Triggers']:>10}")
        
    print("-" * 60)
    print(f"【Ensemble 適應型智能體】最終淨值: {final_equity:,.2f} | 累積報酬: {roi:.2f}% | 夏普值: {trading_sharpe:.4f}")
    print("-" * 60)
    
    labels = ['隨機客', '動能交易者', '風險厭惡者', '風險狂熱者', '當沖客', '羊群跟風者']
    for idx, name in enumerate(['Random', 'Momentum', 'RiskAverse', 'Risk', 'Day', 'Herding']):
        opp_hist = opp_trade_histories[name]
        f_eq = opp_hist[-1]
        o_roi = (f_eq - initial_cap) / initial_cap * 100
        o_sharpe = calculate_sharpe_ratio(opp_hist)
        print(f"【{name} ({labels[idx]})】 淨值: {f_eq:,.2f} | 報酬: {o_roi:.2f}% | 夏普值: {o_sharpe:.4f}")
        
    print("="*60)
    
    # 畫出走勢圖
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    plot_ensemble_performance(total_trade_history, opp_trade_histories)

if __name__ == "__main__":
    run_quantitative_ensemble_system()
