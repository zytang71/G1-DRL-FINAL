import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 偵測是否有 GPU 可用，否則回退至 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算裝置: {device}")

# 1. 定義計畫書中的 7 層 CNN 架構 [cite: 1, 9]
class MarketCNN(nn.Module):
    def __init__(self):
        super(MarketCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=0), nn.ReLU(),  # 7x7
            nn.Conv2d(64, 128, 3, padding=0), nn.ReLU(), # 5x5
            nn.Conv2d(128, 128, 3, padding=0), nn.ReLU(),# 3x3
            nn.Conv2d(128, 256, 3, padding=0), nn.ReLU(),# 1x1
            nn.Conv2d(256, 256, 1, padding=0), nn.ReLU() # 1x1
        )
        self.fc = nn.Linear(256, 8) # 輸出計畫書要求的 8 維空間特徵 

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def run_precalculation(csv_path='market_data.csv'):
    # --- 步驟 A: 讀取與預處理 CSV ---
    df = pd.read_csv(csv_path)
    
    # 根據你的 CSV 截圖，排除 'date' 並保留 12 個特徵 (5價格+7指標)
    raw_features = df.drop(columns=['date']).values 
    
    # 正規化至 [0, 1] [cite: 8, 9]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(raw_features)
    
    # 重構為 9x9x12 市場熱圖影像 [cite: 8]
    num_days = scaled_features.shape[0]
    reshaped_data = scaled_features.reshape(num_days, 12, 1, 1)
    image_data = np.tile(reshaped_data, (1, 1, 9, 9))
    
    # --- 步驟 B: 將資料移至 GPU ---
    input_tensor = torch.from_numpy(image_data).float().to(device)
    
    # --- 步驟 C: 將模型移至 GPU ---
    model = MarketCNN().to(device)
    model.eval()
    
    print(f"正在 GPU 上處理 {num_days} 天的數據...")
    
    with torch.no_grad():
        # 在 GPU 上執行並行卷積運算
        all_spatial_features = model(input_tensor)
        
    # --- 步驟 D: 將結果移回 CPU 並存檔 ---
    # 存檔前建議移回 cpu，以確保未來在沒有 GPU 的環境也能讀取
    torch.save(all_spatial_features.cpu(), 'cnn_features_all.pt')
    print(f"成功！已產出 8 維空間特徵並存至 cnn_features_all.pt")

if __name__ == "__main__":
    # 請確保檔案名稱正確
    run_precalculation('0050_tw.csv')