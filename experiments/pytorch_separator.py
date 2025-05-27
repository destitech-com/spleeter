#!/usr/bin/env python3
"""
PyTorch-based audio separator that exactly mimics original Spleeter logic.

This script uses PyTorch models converted from the original TensorFlow checkpoints
to perform audio source separation.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import librosa

# Import the model architecture from the conversion script
from convert_models_to_pytorch import UNetSingleInstrument


def pad_and_partition(tensor: torch.Tensor, segment_len: int) -> torch.Tensor:
    """
    Pad and partition a tensor into segments of len `segment_len`
    along the first dimension.
    """
    tensor_size = tensor.shape[0] % segment_len
    pad_size = (segment_len - tensor_size) % segment_len
    
    if pad_size > 0:
        padding = torch.zeros(pad_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
        padded = torch.cat([tensor, padding], dim=0)
    else:
        padded = tensor
    
    split = (padded.shape[0] + segment_len - 1) // segment_len
    return padded.view(split, segment_len, *padded.shape[1:])


def pad_and_reshape(instr_spec: torch.Tensor, frame_length: int, F: int) -> torch.Tensor:
    """Pad and reshape spectrogram to match expected dimensions."""
    spec_shape = instr_spec.shape
    extension_row = torch.zeros(spec_shape[0], spec_shape[1], 1, spec_shape[-1], 
                               device=instr_spec.device, dtype=instr_spec.dtype)
    n_extra_row = (frame_length) // 2 + 1 - F
    extension = extension_row.repeat(1, 1, n_extra_row, 1)
    extended_spec = torch.cat([instr_spec, extension], dim=2)
    old_shape = extended_spec.shape
    new_shape = (old_shape[0] * old_shape[1], *old_shape[2:])
    processed_instr_spec = extended_spec.view(new_shape)
    return processed_instr_spec


class PyTorchAudioSeparator:
    """PyTorch-based audio separator matching original Spleeter exactly."""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cpu'):
        """Initialize the separator."""
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device(device)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config["sample_rate"]
        self.frame_length = self.config["frame_length"]
        self.frame_step = self.config["frame_step"]
        self.F = self.config["F"]
        self.T = self.config["T"]
        self.n_channels = self.config["n_channels"]
        self.instruments = self.config["instruments"]
        
        # Key Spleeter parameters
        self.separation_exponent = self.config.get("separation_exponent", 2)
        self.mask_extension = self.config.get("mask_extension", "zeros")
        self.epsilon = 1e-10
        self.window_compensation_factor = 2.0 / 3.0
        
        # Load models
        self.models = {}
        self._load_models()
        
        print(f"Loaded {len(self.models)} instrument models: {list(self.models.keys())}")
        print(f"Using separation_exponent: {self.separation_exponent}")
        print(f"Using mask_extension: {self.mask_extension}")
        print(f"Using device: {self.device}")
    
    def _load_models(self):
        """Load all instrument models."""
        for instrument in self.instruments:
            model_file = os.path.join(self.model_path, f"{instrument}_model.pth")
            if os.path.exists(model_file):
                # Create model and load weights
                model = UNetSingleInstrument(instrument)
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.to(self.device)
                model.eval()  # Set to evaluation mode
                self.models[instrument] = model
                print(f"Loaded PyTorch model for {instrument}")
            else:
                print(f"Warning: Model file not found for {instrument}: {model_file}")
    
    def _stft(self, waveform: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Short-Time Fourier Transform."""
        # Convert to torch tensor
        waveform = torch.from_numpy(waveform).float().to(self.device)
        
        # Pad input with a frame of zeros
        padding = torch.zeros(self.frame_length, self.n_channels, device=self.device)
        waveform = torch.cat([padding, waveform], dim=0)
        
        # Compute STFT
        # PyTorch STFT expects (batch, time) or (time,) for mono
        # We need to transpose to (channels, time) then process each channel
        stft_list = []
        for ch in range(self.n_channels):
            stft_ch = torch.stft(
                waveform[:, ch],
                n_fft=self.frame_length,
                hop_length=self.frame_step,
                window=torch.hann_window(self.frame_length, device=self.device),
                return_complex=True,
                pad_mode='constant'
            )
            stft_list.append(stft_ch)
        
        # Stack channels: (freq, time, channels)
        stft_feature = torch.stack(stft_list, dim=-1)
        
        # Transpose to match TensorFlow format: (time, freq, channels)
        stft_feature = stft_feature.permute(1, 0, 2)
        
        # Compute magnitude spectrogram
        mix_spectrogram = torch.abs(pad_and_partition(stft_feature, self.T))[:, :, :self.F, :]
        
        return stft_feature, mix_spectrogram
    
    def _inverse_stft(self, stft_t: torch.Tensor, time_crop: Optional[int] = None) -> torch.Tensor:
        """
        Inverse STFT to reconstruct waveform.
        """
        # stft_t shape: (time, freq, channels)
        # PyTorch istft expects (batch, freq, time) or (freq, time) for each channel
        
        waveforms = []
        for ch in range(self.n_channels):
            # Extract channel and transpose to (freq, time)
            stft_ch = stft_t[:, :, ch].permute(1, 0)
            
            # Inverse STFT
            waveform_ch = torch.istft(
                stft_ch,
                n_fft=self.frame_length,
                hop_length=self.frame_step,
                window=torch.hann_window(self.frame_length, device=self.device),
                length=None
            )
            waveforms.append(waveform_ch)
        
        # Stack channels and transpose to (time, channels)
        inversed = torch.stack(waveforms, dim=-1) * self.window_compensation_factor
        
        if time_crop is None:
            time_crop = self.waveform_length
        
        # Remove padding and crop to original length
        return inversed[self.frame_length:self.frame_length + time_crop, :]
    
    def _extend_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extend mask from reduced number of frequency bins to
        the number of frequency bins in the STFT.
        """
        extension = self.mask_extension
        
        if extension == "average":
            extension_row = torch.mean(mask, dim=2, keepdim=True)
        elif extension == "zeros":
            mask_shape = mask.shape
            extension_row = torch.zeros(mask_shape[0], mask_shape[1], 1, mask_shape[-1], 
                                      device=mask.device, dtype=mask.dtype)
        else:
            raise ValueError(f"Invalid mask_extension parameter {extension}")
        
        n_extra_row = self.frame_length // 2 + 1 - self.F
        extension = extension_row.repeat(1, 1, n_extra_row, 1)
        return torch.cat([mask, extension], dim=2)
    
    def separate_waveform(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate waveform exactly as original Spleeter pipeline.
        """
        self.waveform_length = waveform.shape[0]
        print(f"Separating audio of shape {waveform.shape}")
        
        with torch.no_grad():  # Disable gradients for inference
            # Get STFT
            stft_feature, spectrogram_feature = self._stft(waveform)
            
            # Run models
            model_outputs = {}
            for instrument, model in self.models.items():
                # Input spectrogram_feature shape: (segments, time, freq, channels)
                # Model handles the format conversion internally
                output = model(spectrogram_feature)
                
                model_outputs[instrument] = output
                print(f"Model output shape: {output.shape}")
            
            # Apply separation logic (same as Keras version)
            output_dict = model_outputs
            separation_exponent = self.separation_exponent
            
            # Compute output sum for normalization
            output_sum = torch.zeros_like(list(output_dict.values())[0])
            for output in output_dict.values():
                output_sum += output ** separation_exponent
            output_sum += self.epsilon
            
            # Compute masks
            masks = {}
            for instrument in self.instruments:
                output = output_dict[instrument]
                # Compute mask
                instrument_mask = (
                    output ** separation_exponent + (self.epsilon / len(output_dict))
                ) / output_sum
                
                # Extend mask
                instrument_mask = self._extend_mask(instrument_mask)
                
                # Stack back mask
                old_shape = instrument_mask.shape
                new_shape = (old_shape[0] * old_shape[1], *old_shape[2:])
                instrument_mask = instrument_mask.view(new_shape)
                
                # Remove padded part (for mask having the same size as STFT)
                instrument_mask = instrument_mask[:stft_feature.shape[0], ...]
                masks[instrument] = instrument_mask
            
            # Apply masks to STFT
            masked_stfts = {}
            for instrument, mask in masks.items():
                # Convert mask to complex for multiplication
                mask_complex = mask.to(dtype=stft_feature.dtype)
                masked_stfts[instrument] = mask_complex * stft_feature
            
            # Inverse STFT to get waveforms
            output_waveform = {}
            for instrument, stft_data in masked_stfts.items():
                waveform_tensor = self._inverse_stft(stft_data)
                output_waveform[instrument] = waveform_tensor.cpu().numpy()
        
        return output_waveform
    
    def separate(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Separate audio file into individual instruments."""
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        
        # Ensure stereo
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform], axis=0)
        
        # Transpose to (time, channels)
        if waveform.shape[0] == 2:
            waveform = waveform.T
        
        print(f"{audio_path} waveform shape: {waveform.shape}, {(waveform.min(), waveform.max())}")
        return self.separate_waveform(waveform)
    
    def save_separated_audio(self, separated_waveforms: Dict[str, np.ndarray], 
                           output_dir: str, filename_prefix: str = "separated"):
        """Save separated waveforms to audio files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for instrument, waveform in separated_waveforms.items():
            # Transpose back to (channels, time) for soundfile
            if waveform.shape[-1] == 2:
                waveform_save = waveform.T
            else:
                waveform_save = waveform
            
            output_path = os.path.join(output_dir, f"{filename_prefix}_{instrument}.wav")
            import soundfile as sf
            sf.write(output_path, waveform_save.T, self.sample_rate)
            print(f"Saved {instrument} to {output_path}")


def demo_pytorch_separation():
    """Demo function showing PyTorch separation."""
    
    # Test 2-stems separation
    model_dir_2stems = "converted_models_pytorch/2stems"
    config_file_2stems = os.path.join(model_dir_2stems, "config.json")
    
    if os.path.exists(config_file_2stems):
        print("Testing PyTorch 2-stems separation...")
        
        # Add missing parameters to config
        with open(config_file_2stems, 'r') as f:
            config = json.load(f)
        
        if "separation_exponent" not in config:
            config["separation_exponent"] = 2
        if "mask_extension" not in config:
            config["mask_extension"] = "zeros"
        
        with open(config_file_2stems, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Choose device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize separator
        separator = PyTorchAudioSeparator(model_dir_2stems, config_file_2stems, device=device)
        
        # Test with example audio if it exists
        audio_file = "8Mile.wav"
        if os.path.exists(audio_file):
            print(f"Separating {audio_file}...")
            
            # Perform separation
            separated = separator.separate(audio_file)
            
            # Save results
            separator.save_separated_audio(separated, "output_pytorch", "demo_pytorch")
            
            print("PyTorch separation completed successfully!")
            print(f"Separated instruments: {list(separated.keys())}")
            for instrument, waveform in separated.items():
                print(f"  {instrument}: {waveform.shape}")
        else:
            print(f"Audio file {audio_file} not found for testing")
    else:
        print("Converted PyTorch models not found. Please run the conversion script first.")


if __name__ == "__main__":
    demo_pytorch_separation() 