import os
import torchaudio
from torch.utils.data import Dataset


class VCTKDEMANDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.clean_files = sorted(os.listdir(
            os.path.join(root_dir, 'train', 'clean')))
        self.noisy_files = sorted(os.listdir(
            os.path.join(root_dir, 'train', 'noisy')))

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = os.path.join(
            self.root_dir, 'train', 'clean', self.clean_files[idx])
        noisy_path = os.path.join(
            self.root_dir, 'train', 'noisy', self.noisy_files[idx])

        clean_waveform, _ = torchaudio.load(clean_path)
        noisy_waveform, _ = torchaudio.load(noisy_path)

        if self.transform:
            clean_waveform = self.transform(clean_waveform)
            noisy_waveform = self.transform(noisy_waveform)

        return noisy_waveform, clean_waveform
    
    
class THCHS30Dataset(Dataset):
    def __init__(self, 
                 noisy_files_path,
                 clean_files_path, 
                 transform=None):
        self.transform = transform
        # 只收集 .wav 和 .mp3 檔案
        noisy_files = sorted([f for f in os.listdir(noisy_files_path)
                                 if f.endswith(('.wav', '.mp3'))])
        clean_files = sorted([f for f in os.listdir(clean_files_path)
                                 if f.endswith(('.wav', '.mp3'))])
        self.noisy_files = [ os.path.join(noisy_files_path,p) for p in noisy_files ]
        self.clean_files = [ os.path.join(clean_files_path,p) for p in clean_files ]
        
        assert len(self.noisy_files) == len(self.clean_files)

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_files[idx]

        # print(f"Attempting to load: {noisy_path}")
        assert os.path.exists(noisy_path), f"File not found: {noisy_path}"

        noisy_waveform, _ = torchaudio.load(noisy_path)
        clean_waveform, _ = torchaudio.load(clean_path)

        if self.transform:
            noisy_waveform = self.transform(noisy_waveform)
            clean_waveform = self.transform(clean_waveform)

        return noisy_waveform, clean_waveform