import torch

def check_feature_shapes():
    try:
        # 載入空間特徵 (CNN)
        cnn_feat = torch.load('cnn_features_all.pt')
        # 載入時間特徵 (LSTM)
        lstm_feat = torch.load('lstm_features_all.pt')
        
        print(f"--- 特徵維度檢查 ---")
        print(f"CNN 空間特徵維度 (cnn_features_all.pt): {cnn_feat.shape}")
        print(f"LSTM 時間特徵維度 (lstm_features_all.pt): {lstm_feat.shape}")
        
        # 檢查兩者天數是否對齊
        if cnn_feat.shape[0] == lstm_feat.shape[0]:
            print(f"✅ 狀態：天數對齊，共 {cnn_feat.shape[0]} 天。")
        else:
            print(f"❌ 警告：天數不一致！請檢查資料處理邏輯。")
            
    except FileNotFoundError as e:
        print(f"找不到檔案：{e.filename}，請確認是否已執行預處理腳本。")

if __name__ == "__main__":
    check_feature_shapes()