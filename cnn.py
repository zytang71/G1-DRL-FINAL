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

# 1. 定義 7 層 CNN 架構
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
        self.fc = nn.Linear(256, 8) 

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def process_single_file(csv_path, model):
    """處理單一 CSV 檔案並儲存結果"""
    # 取得檔案名稱（不含副檔名），例如 '0050_tw'
    file_stem = Path(csv_path).stem
    
    # 建立目標資料夾（如果不存在）
    output_dir = os.path.join("features", file_stem)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- 正在處理: {file_stem} ---")
    
    # --- 步驟 A: 讀取與預處理 ---
    try:
        df = pd.read_csv(csv_path)
        # 排除 'date' 並保留 12 個特徵
        raw_features = df.drop(columns=['date']).values 
        
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(raw_features)
        
        num_days = scaled_features.shape[0]
        # 重構為 9x9x12 市場熱圖影像
        reshaped_data = scaled_features.reshape(num_days, 12, 1, 1)
        image_data = np.tile(reshaped_data, (1, 1, 9, 9))
        
        # --- 步驟 B & C: GPU 運算 ---
        input_tensor = torch.from_numpy(image_data).float().to(device)
        
        with torch.no_grad():
            all_spatial_features = model(input_tensor)
            
        # --- 步驟 D: 存檔 ---
        save_path = os.path.join(output_dir, 'cnn_features_all.pt')
        torch.save(all_spatial_features.cpu(), save_path)
        print(f"成功！已產出 {num_days} 天的特徵並存至 {save_path}")
        
    except Exception as e:
        print(f"處理檔案 {csv_path} 時發生錯誤: {e}")

def run_batch_precalculation(input_folder='data'):
    # 初始化模型並移至 GPU
    model = MarketCNN().to(device)
    model.eval()

    # 取得 data 資料夾下所有的 CSV 檔案
    data_path = Path(input_folder)
    csv_files = list(data_path.glob('*.csv'))

    if not csv_files:
        print(f"在 '{input_folder}' 資料夾中找不到任何 CSV 檔案。")
        return

    print(f"共找到 {len(csv_files)} 個檔案，開始批次處理...")

    for file_path in csv_files:
        process_single_file(file_path, model)

    print("\n所有任務已完成！")

if __name__ == "__main__":
    # 確保 'data' 資料夾存在
    run_batch_precalculation('data')