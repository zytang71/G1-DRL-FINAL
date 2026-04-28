import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 偵測是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算裝置: {device}")

# 1. 定義計畫書中的 LSTM 架構 (含 12 個特徵輸入)
class MarketLSTM(nn.Module):
    def __init__(self):
        super(MarketLSTM, self).__init__()
        # input_size=12 對應 5種價格 + 7種指標
        self.lstm = nn.LSTM(input_size=12, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 8) # 輸出計畫書要求的 8 維時間特徵 

    def forward(self, x):
        # x shape: (Batch, Seq_Len, 12)
        _, (hn, _) = self.lstm(x)
        # 取最後一個時間步的隱狀態 (Hidden State)
        return self.fc(hn[-1])

def run_precalculation(csv_path='market_data.csv'):
    # --- 步驟 A: 讀取 CSV 並進行正規化 ---
    df = pd.read_csv(csv_path)
    
    # 排除日期，保留 12 個特徵
    raw_features = df.drop(columns=['date']).values 
    
    # 正規化至 [0, 1] 以利 LSTM 收斂
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(raw_features) # 維度: [N, 12]
    
    # --- 步驟 B: 初始化模型並移至 GPU ---
    model = MarketLSTM().to(device)
    model.eval()
    
    num_days = scaled_features.shape[0]
    all_temporal_features = []
    
    print(f"正在 GPU 上執行逐日滑動窗口提取 (共 {num_days} 天)...")

    # --- 步驟 C: 執行滑動窗口特徵提取 ---
    with torch.no_grad():
        for i in range(num_days):
            # 建立 12 天滑動窗口 
            # 若天數不足 12 天，則取從頭開始到當下的所有資料
            start_idx = max(0, i - 11)
            seq = scaled_features[start_idx : i+1]
            
            # 轉換為張量並移至 GPU (Batch, Seq_Len, Features)
            seq_tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
            
            # 提取 8 維時間特徵
            feat = model(seq_tensor)
            all_temporal_features.append(feat.cpu()) # 先收回到 CPU 節省顯存

    # --- 步驟 D: 合併與存檔 ---
    final_temporal_features = torch.cat(all_temporal_features, dim=0)
    torch.save(final_temporal_features, 'lstm_features_all.pt')
    print(f"成功！已產出 {final_temporal_features.shape[0]} 天的 LSTM 特徵，並存至 lstm_features_all.pt")

if __name__ == "__main__":
    # 請確保檔案名稱正確 (例如你上傳的 csv)
    run_precalculation('0050_tw.csv')