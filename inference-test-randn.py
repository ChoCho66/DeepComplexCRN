from dc_crn import DCCRN
import torch
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = DCCRN().to(device).eval()

# 載入預訓練權重
checkpoint_path = "checkpoints/dccrn_trained_on_vctk_epoch200.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)  # 讀取檔案
model.load_state_dict(checkpoint["model"])  # 只讀取 model 權重
print(f"Loaded model weights from {checkpoint_path}")


for i in range(3):
    N = random.randint(300000, 500000)  # 生成不同的 N 值
    input_tensor = torch.randn(1, 1, N).to(device)  # 生成 shape (1,1,N) 的 tensor
    _, output_tensor = model(input_tensor)  # 前向傳遞
    # print(input_tensor.shape, output_tensor.shape)
    
    output_tensor = output_tensor.unsqueeze(1)
    
    padding_length = input_tensor.shape[-1] - output_tensor.shape[-1]
    # 填充 output_tensor 到目标形状
    if padding_length > 0:
        output_tensor = torch.nn.functional.pad(output_tensor, (0, padding_length), mode="constant", value=0)

    print(f'Input shape: {input_tensor.shape}, Output shape: {output_tensor.shape}')

    # print(input_tensor[0,0,padding_length:])
    # print(output_tensor[0,0,padding_length:])
    print()