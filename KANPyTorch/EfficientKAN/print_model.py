import torch

# 設定模型權重檔案路徑
model_path = "model/kan_multiple_weights.pth"

# 載入模型權重（使用 weights_only=True 來確保未來相容性）
try:
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    print("成功載入權重檔案！")
except TypeError:
    print("WARNING: PyTorch 版本不支援 weights_only，使用預設方式載入。")
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# 輸出權重名稱與形狀
print("\n模型權重列表:")
with open("weight.txt", "w") as f:
    for name, param in model_state_dict.items():
        f.write(f"{name}: {param.shape}")
        for i in param:
            f.write(str(i))