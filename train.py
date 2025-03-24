import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dc_crn import DCCRN
from loss import SISNRLoss
from dataset import VCTKDEMANDDataset
from tqdm import tqdm
import torchaudio

# Configuration parameters
# batch_size = 1
batch_size = 16
# batch_size = 32  
# batch_size = 32 顯存會爆 
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 472.00 MiB. GPU 0 has a total capacity of 23.70 GiB of which 114.56 MiB is free. Including non-PyTorch memory, this process has 23.58 GiB memory in use. Of the allocated memory 18.26 GiB is allocated by PyTorch, and 3.69 GiB is reserved by PyTorch but unallocated.

num_epochs = 200
# num_epochs = 1
learning_rate = 0.001
checkpoint_dir = 'checkpoints'
data_dir = '/disk4/chocho/_datas/VCTK_DEMAND16k'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

os.makedirs(checkpoint_dir, exist_ok=True)


def collate_fn(batch):
    """Pads batch of variable length Tensors."""
    # Find the longest sequence in the batch
    max_length = max([x[0].shape[1] for x in batch])

    # Pad sequences to the max length
    padded_noisy = torch.zeros(len(batch), 1, max_length)
    padded_clean = torch.zeros(len(batch), 1, max_length)

    for i, (noisy, clean) in enumerate(batch):
        length = noisy.shape[1]
        padded_noisy[i, 0, :length] = noisy
        padded_clean[i, 0, :length] = clean

    return padded_noisy, padded_clean


# Instantiate the dataset and DataLoader
train_dataset = VCTKDEMANDDataset(root_dir=data_dir)
# collate_fn 會把 dim = 2 的地方填滿
# 也就是這個 batch 的每個音訊會填充到這 batch 最長音訊的長度
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Instantiate the model and loss function
model = DCCRN().to(device).train()
loss_func = SISNRLoss().to(device)

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with early stopping based on loss threshold
# loss_threshold = 0.5  # Desired loss threshold
loss_threshold = -10000.5  # Desired loss threshold

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for noisy_waveform, clean_waveform in tqdm(train_loader):
        # print("---------------------------------------------")
        # print(noisy_waveform.shape, clean_waveform.shape)
        noisy_waveform = noisy_waveform.to(device)
        clean_waveform = clean_waveform.to(device)

        optimizer.zero_grad()

        # Forward pass
        _, outputs = model(noisy_waveform)
        # print(_.shape, outputs.shape)

        outputs = outputs.unsqueeze(1)

        # 计算所需的填充长度
        padding_length = clean_waveform.size(-1) - outputs.size(-1)

        # 填充 outputs 到目标形状
        if padding_length > 0:
            outputs = torch.nn.functional.pad(outputs, (0, padding_length), mode="constant", value=0)
        loss = loss_func(outputs, clean_waveform)
        # print(outputs.shape, clean_waveform.shape)
        # print("---------------------------------------------")
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # print(outputs[0].max(),outputs[0].min())
    # print(outputs[0].shape)
    # torchaudio.save("output.wav", outputs[0].to("cpu"), 16000)
    # torchaudio.save("clean_waveform.wav", clean_waveform[0].to("cpu"), 16000)

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    if avg_loss <= loss_threshold:
        print(f"Stopping training as loss is below {loss_threshold}")
        break 

print(clean_waveform[0].shape,noisy_waveform[0].shape, outputs[0].shape)
torchaudio.save("01clean_waveform.wav", clean_waveform[0].to("cpu"), 16000)
torchaudio.save("02noisy_waveform.wav", noisy_waveform[0].to("cpu"), 16000)
torchaudio.save("03output.wav", outputs[0].to("cpu"), 16000)

torch.save({'model': model.state_dict()}, os.path.join(
    checkpoint_dir, f'dccrn_trained_on_vctk_epoch{epoch+1}.pt'))


print("Training complete and model saved")
