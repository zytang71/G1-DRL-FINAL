import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 偵測是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算裝置: {device}")

# 1. 定義 LSTM 架構
class MarketLSTM(nn.Module):
    def __init__(self):
        super(MarketLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 8) 

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 12)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def process_single_file(csv_path, model):
    """處理單一 CSV 檔案並執行 LSTM 滑動窗口特徵提取"""
    file_stem = Path(csv_path).stem
    
    # 建立目標資料夾（例如：0050_tw/）
    output_dir = os.path.join("features", file_stem)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- 正在處理: {file_stem} ---")
    
    try:
        # --- 步驟 A: 讀取與預處理 ---
        df = pd.read_csv(csv_path)
        raw_features = df.drop(columns=['date']).values 
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(raw_features)
        
        num_days = scaled_features.shape[0]
        all_temporal_features = []
        
        # --- 步驟 B: 執行滑動窗口特徵提取 ---
        with torch.no_grad():
            for i in range(num_days):
                # 12 天滑動窗口
                start_idx = max(0, i - 11)
                seq = scaled_features[start_idx : i+1]
                
                # 轉換為張量 (Batch=1, Seq_Len, Features=12)
                seq_tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
                
                feat = model(seq_tensor)
                all_temporal_features.append(feat.cpu())
        
        # --- 步驟 C: 合併與存檔 ---
        final_temporal_features = torch.cat(all_temporal_features, dim=0)
        save_path = os.path.join(output_dir, 'lstm_features_all.pt')
        torch.save(final_temporal_features, save_path)
        
        print(f"成功！已產出 {final_temporal_features.shape[0]} 天的特徵並存至 {save_path}")
        
    except Exception as e:
        print(f"處理檔案 {csv_path} 時發生錯誤: {e}")

def run_batch_precalculation(input_folder='data'):
    # 初始化模型
    model = MarketLSTM().to(device)
    model.eval()

    # 取得資料夾內所有 CSV
    data_path = Path(input_folder)
    csv_files = list(data_path.glob('*.csv'))

    if not csv_files:
        print(f"在 '{input_folder}' 中找不到 CSV 檔案。")
        return

    print(f"找到 {len(csv_files)} 個標的，開始執行 LSTM 特徵提取...")

    for file_path in csv_files:
        process_single_file(file_path, model)

    print("\n所有 LSTM 特徵提取任務已完成！")

if __name__ == "__main__":
    # 預設讀取 data 資料夾
    run_batch_precalculation('data')