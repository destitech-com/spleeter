#!/usr/bin/env python3
"""
Corrected Keras-based audio separator that exactly mimics original Spleeter logic.

The key insight: the original Spleeter treats the model outputs (mask * input) as 
magnitude estimates for ratio mask computation.
"""

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.signal import hann_window, inverse_stft, stft  # type: ignore

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from typing import Dict, List, Optional, Tuple
import librosa

from typing import Any, Dict, Optional, Tuple

def pad_and_partition(tensor: tf.Tensor, segment_len: int) -> tf.Tensor:
    """
    Pad and partition a tensor into segment of len `segment_len`
    along the first dimension. The tensor is padded with 0 in order
    to ensure that the first dimension is a multiple of `segment_len`.

    Examples:
    ```python
    >>> tensor = [[1, 2, 3], [4, 5, 6]]
    >>> segment_len = 2
    >>> pad_and_partition(tensor, segment_len)
    [[[1, 2], [4, 5]], [[3, 0], [6, 0]]]
    ````

    Parameters:
        tensor (tf.Tensor):
            Tensor of known fixed rank
        segment_len (int):
            Segment length.

    Returns:
        tf.Tensor:
            Padded and partitioned tensor.
    """
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(tensor, [[0, pad_size]] + [[0, 0]] * (len(tensor.shape) - 1))
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(
        padded, tf.concat([[split, segment_len], tf.shape(padded)[1:]], axis=0)
    )


def pad_and_reshape(instr_spec, frame_length, F) -> Any:
    spec_shape = tf.shape(instr_spec)
    extension_row = tf.zeros((spec_shape[0], spec_shape[1], 1, spec_shape[-1]))
    n_extra_row = (frame_length) // 2 + 1 - F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    extended_spec = tf.concat([instr_spec, extension], axis=2)
    old_shape = tf.shape(extended_spec)
    new_shape = tf.concat([[old_shape[0] * old_shape[1]], old_shape[2:]], axis=0)
    processed_instr_spec = tf.reshape(extended_spec, new_shape)
    return processed_instr_spec


class CorrectKerasAudioSeparator:
    """Corrected Keras-based audio separator matching original Spleeter exactly."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize the separator."""
        self.model_path = model_path
        self.config_path = config_path
        
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
    
    def _load_models(self):
        """Load all instrument models."""
        for instrument in self.instruments:
            model_file = os.path.join(self.model_path, f"{instrument}_model.keras")
            if os.path.exists(model_file):
                self.models[instrument] = tf.keras.models.load_model(model_file)
                print(f"Loaded model for {instrument}")
            else:
                print(f"Warning: Model file not found for {instrument}: {model_file}")
    
    def _stft(self, waveform: np.ndarray) -> tf.Tensor:
        """Compute Short-Time Fourier Transform."""
        waveform = tf.concat(
            [
                tf.zeros((self.frame_length, self.n_channels)),
                waveform,
            ],
            0,
        )
        stft_feature = tf.transpose(
            stft(
                tf.transpose(waveform),
                self.frame_length,
                self.frame_step,
                window_fn=lambda frame_length, dtype: (
                    hann_window(frame_length, periodic=True, dtype=dtype)
                ),
                pad_end=True,
            ),
            perm=[1, 2, 0],
        )
        mix_stft = stft_feature
        mix_spectrogram = tf.abs(
            pad_and_partition(mix_stft, self.T)
        )[:, :, : self.F, :]

        return mix_stft, mix_spectrogram
    

    def _inverse_stft(
        self, stft_t: tf.Tensor, time_crop: Optional[Any] = None
    ) -> tf.Tensor:
        """
        Inverse and reshape the given STFT

        Parameters:
            stft_t (tf.Tensor):
                Input STFT.
            time_crop (Optional[Any]):
                Time cropping.

        Returns:
            tf.Tensor:
                Inverse STFT (waveform).
        """
        inversed = (
            inverse_stft(
                tf.transpose(stft_t, perm=[2, 0, 1]),
                self.frame_length,
                self.frame_step,
                window_fn=lambda frame_length, dtype: (
                    hann_window(frame_length, periodic=True, dtype=dtype)
                ),
            )
            * self.window_compensation_factor
        )
        reshaped = tf.transpose(inversed)
        if time_crop is None:
            time_crop = self.waveform_length
        return reshaped[self.frame_length : self.frame_length + time_crop, :]


    def _extend_mask(self, mask: tf.Tensor) -> tf.Tensor:
        """
        Extend mask, from reduced number of frequency bin to
        the number of frequency bin in the STFT.

        Parameters:
            mask (tf.Tensor):
                Restricted mask.

        Returns:
            tf.Tensor:
                Extended mask

        Raises:
            ValueError:
                If invalid mask_extension parameter is set.
        """
        extension = self.mask_extension
        # Extend with average
        # (dispatch according to energy in the processed band)
        if extension == "average":
            extension_row = tf.reduce_mean(mask, axis=2, keepdims=True)
        # Extend with 0
        # (avoid extension artifacts but not conservative separation)
        elif extension == "zeros":
            mask_shape = tf.shape(mask)
            extension_row = tf.zeros((mask_shape[0], mask_shape[1], 1, mask_shape[-1]))
        else:
            raise ValueError(f"Invalid mask_extension parameter {extension}")
        n_extra_row = self.frame_length // 2 + 1 - self.F
        extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
        return tf.concat([mask, extension], axis=2)



    def separate_waveform(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate waveform exactly as original Spleeter pipeline.
        """
        self.waveform_length = waveform.shape[0]
        print(f"Separating audio of shape {waveform.shape}")
        
        # Get STFT
        stft_feature, spectrogram_feature = self._stft(waveform)

        
        # write tensor to pickle as numpy array
        import pickle
        with open('input_tensor_spectrogram.pkl', 'wb') as f:
            pickle.dump(spectrogram_feature.numpy(), f)

        model_outputs = {}
        for instrument, model in self.models.items():
            print(f"Input shape {instrument}: {spectrogram_feature.shape} ({spectrogram_feature.min()}, {spectrogram_feature.max()})")
            model_outputs[instrument] = model(spectrogram_feature)
            print(f"Model output shape {instrument}: {model_outputs[instrument].shape} ({model_outputs[instrument].min()}, {model_outputs[instrument].max()})")

        
        output_dict = model_outputs
        stft_feature = stft_feature
        separation_exponent = self.separation_exponent
        output_sum = (
            tf.reduce_sum(
                [e ** separation_exponent for e in output_dict.values()], axis=0
            )
            + self.epsilon
        )
        masks = {}
        for instrument in self.instruments:
            output = output_dict[instrument]
            # Compute mask with the model.
            instrument_mask = (
                output ** separation_exponent + (self.epsilon / len(output_dict))
            ) / output_sum
            # Extend mask;
            instrument_mask = self._extend_mask(instrument_mask)
            # Stack back mask.
            old_shape = tf.shape(instrument_mask)
            new_shape = tf.concat(
                [[old_shape[0] * old_shape[1]], old_shape[2:]], axis=0
            )
            instrument_mask = tf.reshape(instrument_mask, new_shape)
            # Remove padded part (for mask having the same size as STFT);

            instrument_mask = instrument_mask[: tf.shape(stft_feature)[0], ...]
            masks[instrument] = instrument_mask


        masked_stfts = {}
        for instrument, mask in masks.items():
            masked_stfts[instrument] = tf.cast(mask, dtype=tf.complex64) * stft_feature


        output_waveform = {}
        for instrument, stft_data in masked_stfts.items():
            output_waveform[instrument] = self._inverse_stft(stft_data)
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


def demo_correct_separation():
    """Demo function showing the correct separation."""
    
    # Test 2-stems separation
    model_dir_2stems = "converted_models/2stems"
    config_file_2stems = os.path.join(model_dir_2stems, "config.json")
    
    if os.path.exists(config_file_2stems):
        print("Testing CORRECT 2-stems separation...")
        
        # Add missing parameters to config
        with open(config_file_2stems, 'r') as f:
            config = json.load(f)
        
        if "separation_exponent" not in config:
            config["separation_exponent"] = 2
        if "mask_extension" not in config:
            config["mask_extension"] = "zeros"
        
        with open(config_file_2stems, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize separator
        separator = CorrectKerasAudioSeparator(model_dir_2stems, config_file_2stems)
        
        # Test with example audio if it exists
        audio_file = "audio_example.mp3"
        if os.path.exists(audio_file):
            print(f"Separating {audio_file}...")
            
            # Perform separation
            separated = separator.separate(audio_file)
            
            # Save results
            separator.save_separated_audio(separated, "output_correct", "demo_correct")
            
            print("CORRECT separation completed successfully!")
            print(f"Separated instruments: {list(separated.keys())}")
            for instrument, waveform in separated.items():
                print(f"  {instrument}: {waveform.shape}")
        else:
            print(f"Audio file {audio_file} not found for testing")
    else:
        print("Converted models not found. Please run the conversion script first.")


if __name__ == "__main__":
    demo_correct_separation() 