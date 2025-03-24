from dc_crn import DCCRN
import torch
import os
import soundfile as sf  # For reading and writing audio files
# import torchaudio  # For audio processing utilities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize model
model = DCCRN().to(device).eval()

# Load pre-trained weights
checkpoint_path = "checkpoints/dccrn_trained_on_vctk_epoch200.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
print(f"Loaded model weights from {checkpoint_path}")

# Input and output directories
input_folder = "/disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy"  # Replace with your input folder path
output_folder = "DCCRN-output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported audio extensions
audio_extensions = ('.wav', '.flac', '.mp3')

# Process all audio files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(audio_extensions):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Read audio file
        audio, sample_rate = sf.read(input_path)
        
        # Convert to tensor and prepare for model
        audio_tensor = torch.from_numpy(audio).float()
        
        # print(audio_tensor.shape)
        
        # Ensure audio is mono (single channel)
        if audio_tensor.ndim == 2:  # If stereo
            audio_tensor = torch.mean(audio_tensor, dim=1)  # Convert to mono
        
        # Add batch and channel dimensions: [1, 1, length]
        input_tensor = audio_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Process through model
        with torch.no_grad():
            _, output_tensor = model(input_tensor)
        
        # print(output_tensor.shape)
        
        # Remove channel dimension and convert to numpy
        output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
        output_audio = output_tensor.cpu().numpy()
        
        # Pad or truncate output to match input length
        input_length = audio_tensor.shape[-1]
        output_length = output_audio.shape[-1]
        
        if output_length < input_length:
            # Pad with zeros
            padding = input_length - output_length
            output_audio = torch.nn.functional.pad(
                torch.from_numpy(output_audio), 
                (0, padding), 
                mode="constant", 
                value=0
            ).numpy()
        elif output_length > input_length:
            # Truncate to match input length
            output_audio = output_audio[:input_length]
        
        # print(output_audio.shape)
        
        # Write enhanced audio to output file
        sf.write(output_path, output_audio, sample_rate, subtype='FLOAT')
        print(f"Processed: {filename} -> {output_path}")
        # print(input_length, output_audio.shape)

print("Speech enhancement completed for all audio files!")