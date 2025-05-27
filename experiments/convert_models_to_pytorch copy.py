#!/usr/bin/env python3
"""
Script to convert Spleeter TensorFlow 1.x checkpoint models to PyTorch format.

This script loads the pretrained TF 1.x models and converts them to PyTorch models
that can be used with modern PyTorch versions.
"""

import os
import json
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Disable TF2 behavior to work with TF1 checkpoints
tf.compat.v1.disable_v2_behavior()

class UNetSingleInstrument(nn.Module):
    """PyTorch U-Net model for single instrument separation."""
    
    def __init__(self, instrument_name: str):
        super(UNetSingleInstrument, self).__init__()
        self.instrument_name = instrument_name
        
        # Network parameters
        conv_n_filters = [16, 32, 64, 128, 256, 512]
        
        # Note: Input format is (batch, time, freq, channels) to match TensorFlow
        # We'll use permute operations to handle this
        
        # Encoder layers - input channels = 2 (stereo)
        self.encoder_conv1 = nn.Conv2d(2, conv_n_filters[0], kernel_size=5, stride=2, padding=0)
        self.encoder_bn1 = nn.BatchNorm2d(conv_n_filters[0], eps=1e-3, momentum=0.01)
        
        self.encoder_conv2 = nn.Conv2d(conv_n_filters[0], conv_n_filters[1], kernel_size=5, stride=2, padding=0)
        self.encoder_bn2 = nn.BatchNorm2d(conv_n_filters[1], eps=1e-3, momentum=0.01)
        
        self.encoder_conv3 = nn.Conv2d(conv_n_filters[1], conv_n_filters[2], kernel_size=5, stride=2, padding=0)
        self.encoder_bn3 = nn.BatchNorm2d(conv_n_filters[2], eps=1e-3, momentum=0.01)
        
        self.encoder_conv4 = nn.Conv2d(conv_n_filters[2], conv_n_filters[3], kernel_size=5, stride=2, padding=0)
        self.encoder_bn4 = nn.BatchNorm2d(conv_n_filters[3], eps=1e-3, momentum=0.01)
        
        self.encoder_conv5 = nn.Conv2d(conv_n_filters[3], conv_n_filters[4], kernel_size=5, stride=2, padding=0)
        self.encoder_bn5 = nn.BatchNorm2d(conv_n_filters[4], eps=1e-3, momentum=0.01)
        
        self.encoder_conv6 = nn.Conv2d(conv_n_filters[4], conv_n_filters[5], kernel_size=5, stride=2, padding=0)
        # Note: encoder_bn6 is not used in the original model
        
        # Decoder layers
        self.decoder_conv1 = nn.ConvTranspose2d(conv_n_filters[5], conv_n_filters[4], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(conv_n_filters[4], eps=1e-3, momentum=0.01)
        
        self.decoder_conv2 = nn.ConvTranspose2d(conv_n_filters[4] + conv_n_filters[4], conv_n_filters[3], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(conv_n_filters[3], eps=1e-3, momentum=0.01)
        
        self.decoder_conv3 = nn.ConvTranspose2d(conv_n_filters[3] + conv_n_filters[3], conv_n_filters[2], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_bn3 = nn.BatchNorm2d(conv_n_filters[2], eps=1e-3, momentum=0.01)
        
        self.decoder_conv4 = nn.ConvTranspose2d(conv_n_filters[2] + conv_n_filters[2], conv_n_filters[1], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_bn4 = nn.BatchNorm2d(conv_n_filters[1], eps=1e-3, momentum=0.01)
        
        self.decoder_conv5 = nn.ConvTranspose2d(conv_n_filters[1] + conv_n_filters[1], conv_n_filters[0], kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_bn5 = nn.BatchNorm2d(conv_n_filters[0], eps=1e-3, momentum=0.01)
        
        self.decoder_conv6 = nn.ConvTranspose2d(conv_n_filters[0] + conv_n_filters[0], 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_bn6 = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        
        # Output mask layer - matches TF up7 layer with padding="same"
        # TF: Conv2D(2, (4,4), dilation_rate=(2,2), padding='same')
        # For 'same' padding with dilation=2 and kernel=4: padding = (kernel-1)*dilation//2 = 3*2//2 = 3
        self.mask_conv = nn.Conv2d(1, 2, kernel_size=4, padding=3, dilation=2)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch, time, freq, channels) - TensorFlow format
        # Convert to PyTorch format: (batch, channels, time, freq)
        input_tensor = x
        x = x.permute(0, 3, 1, 2)  # (batch, channels, time, freq)
        # print(f"Input shape: {input_tensor.shape} -> {x.shape} ({x.min()}, {x.max()})")
        
        # Encoder path
        # Store direct conv outputs for skip connections
        encoder_conv_outputs = []

        x_pad1 = F.pad(x, (1, 2, 1, 2), mode='constant', value=0)
        e_conv1 = self.encoder_conv1(x_pad1)
        encoder_conv_outputs.append(e_conv1)
        batch1 = self.encoder_bn1(e_conv1)
        rel1 = self.leaky_relu(batch1)
        # print(f"Step1 (conv1): {e_conv1.shape} ({e_conv1.min():.6f}, {e_conv1.max():.6f})")
        # print(f"Step1 (after bn): { batch1.shape} ({batch1.min():.6f}, {batch1.max():.6f})")
        # print(f"Step1 (after bn+relu): {rel1.shape} ({rel1.min():.6f}, {rel1.max():.6f})")
        
        rel1_pad2 = F.pad(rel1, (1, 2, 1, 2), mode='constant', value=0)
        e_conv2 = self.encoder_conv2(rel1_pad2)
        encoder_conv_outputs.append(e_conv2)
        batch2 = self.encoder_bn2(e_conv2)
        rel2 = self.leaky_relu(batch2)
        # print(f"Step2 (conv2): {e_conv2.shape} ({e_conv2.min():.6f}, {e_conv2.max():.6f})")
        # print(f"Step2 (after bn+relu): {rel2.shape} ({rel2.min():.6f}, {rel2.max():.6f})")
        
        rel2_pad3 = F.pad(rel2, (1, 2, 1, 2), mode='constant', value=0)
        e_conv3 = self.encoder_conv3(rel2_pad3)
        encoder_conv_outputs.append(e_conv3)
        batch3 = self.encoder_bn3(e_conv3)
        rel3 = self.leaky_relu(batch3)
        # print(f"Step3 (conv3): {e_conv3.shape} ({e_conv3.min():.6f}, {e_conv3.max():.6f})")
        # print(f"Step3 (after bn+relu): {rel3.shape} ({rel3.min():.6f}, {rel3.max():.6f})")
        
        rel3_pad4 = F.pad(rel3, (1, 2, 1, 2), mode='constant', value=0)
        e_conv4 = self.encoder_conv4(rel3_pad4)
        encoder_conv_outputs.append(e_conv4)
        batch4 = self.encoder_bn4(e_conv4)
        rel4 = self.leaky_relu(batch4)
        # print(f"Step4 (conv4): {e_conv4.shape} ({e_conv4.min():.6f}, {e_conv4.max():.6f})")
        # print(f"Step4 (after bn+relu): {rel4.shape} ({rel4.min():.6f}, {rel4.max():.6f})")
        
        rel4_pad5 = F.pad(rel4, (1, 2, 1, 2), mode='constant', value=0)
        e_conv5 = self.encoder_conv5(rel4_pad5)
        encoder_conv_outputs.append(e_conv5)
        batch5 = self.encoder_bn5(e_conv5)
        rel5 = self.leaky_relu(batch5)
        # print(f"Step5 (conv5): {e_conv5.shape} ({e_conv5.min():.6f}, {e_conv5.max():.6f})")
        # print(f"Step5 (after bn+relu): {rel5.shape} ({rel5.min():.6f}, {rel5.max():.6f})")
        
        rel5_pad6 = F.pad(rel5, (1, 2, 1, 2), mode='constant', value=0)
        e_conv6 = self.encoder_conv6(rel5_pad6) # This is the bottleneck, not used for skip
        # print(f"Step6 (conv6): {e_conv6.shape} ({e_conv6.min():.6f}, {e_conv6.max():.6f})")
        
        # Decoder path with skip connections
        up1 = self.decoder_conv1(e_conv6)
        up1 = self.relu(up1)
        batch7 = self.decoder_bn1(up1)
        drop1 = self.dropout(batch7)
        merge1 = torch.cat([encoder_conv_outputs[4], drop1], dim=1) # Skip from e_conv5 (index 4)
        # print(f"Step7 (batch7): {batch7.shape} ({batch7.min():.6f}, {batch7.max():.6f})")
        # print(f"Step7 (drop1): {drop1.shape} ({drop1.min():.6f}, {drop1.max():.6f})")
        # print(f"Step7 (merge1): {merge1.shape} ({merge1.min():.6f}, {merge1.max():.6f})")
        
        up2 = self.decoder_conv2(merge1)
        up2 = self.relu(up2)
        batch8 = self.decoder_bn2(up2)
        drop2 = self.dropout(batch8)
        merge2 = torch.cat([encoder_conv_outputs[3], drop2], dim=1) # Skip from e_conv4 (index 3)
        # print(f"Step8 (batch8): {batch8.shape} ({batch8.min():.6f}, {batch8.max():.6f})")
        # print(f"Step8 (drop2): {drop2.shape} ({drop2.min():.6f}, {drop2.max():.6f})")
        # print(f"Step8 (merge2): {merge2.shape} ({merge2.min():.6f}, {merge2.max():.6f})")
        
        up3 = self.decoder_conv3(merge2)
        up3 = self.relu(up3)
        batch9 = self.decoder_bn3(up3)
        drop3 = self.dropout(batch9)
        merge3 = torch.cat([encoder_conv_outputs[2], drop3], dim=1) # Skip from e_conv3 (index 2)
        # print(f"Step9: {merge3.shape} ({merge3.min()}, {merge3.max()})")
        
        up4 = self.decoder_conv4(merge3)
        up4 = self.relu(up4)
        batch10 = self.decoder_bn4(up4)
        # Original Keras model uses conv2 output for skip, then batch10 (which is after BN, no dropout)
        # So, use encoder_conv_outputs[1] (e_conv2)
        merge4 = torch.cat([encoder_conv_outputs[1], batch10], dim=1) # Skip from e_conv2 (index 1)
        # print(f"Step10: {merge4.shape} ({merge4.min()}, {merge4.max()})")
        
        up5 = self.decoder_conv5(merge4)
        up5 = self.relu(up5)
        batch11 = self.decoder_bn5(up5)
        # Original Keras model uses conv1 output for skip, then batch11 (which is after BN, no dropout)
        # So, use encoder_conv_outputs[0] (e_conv1)
        merge5 = torch.cat([encoder_conv_outputs[0], batch11], dim=1) # Skip from e_conv1 (index 0)
        # print(f"Step11: {merge5.shape} ({merge5.min()}, {merge5.max()})")
        
        up6 = self.decoder_conv6(merge5)
        up6 = self.relu(up6)
        batch12 = self.decoder_bn6(up6)
        print(f"Step12: {batch12.shape} ({batch12.min()}, {batch12.max()})")
        
        # Output mask
        mask = self.mask_conv(batch12)
        mask = self.sigmoid(mask)
        print(f"Step13: {mask.shape} ({mask.min()}, {mask.max()})")
        
        # Convert mask back to TensorFlow format: (batch, time, freq, channels)
        mask = mask.permute(0, 2, 3, 1)
        print(f"Mask permute: {mask.shape}")
        
        # Apply mask to input
        output = mask * input_tensor
        print(f"output: {output.shape} ({output.min()}, {output.max()})")
        
        return output


class SpleeterPyTorchConverter:
    """Converts Spleeter TF1.x models to PyTorch format."""
    
    def __init__(self):
        """Initialize the converter."""
        self.sample_rate = 44100
        self.frame_length = 4096
        self.frame_step = 1024
        self.T = 512
        self.F = 1024
        self.n_channels = 2
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def load_tf1_weights_detailed(self, checkpoint_path: str) -> Dict[str, np.ndarray]:
        """
        Load all weights from TF1.x checkpoint.
        
        Args:
            checkpoint_path: Path to the TF1.x checkpoint
            
        Returns:
            Dictionary of all variable weights
        """
        print(f"Loading weights from {checkpoint_path}")
        
        tf1_var_values = {}
        
        with tf.compat.v1.Session() as sess:
            # Load the checkpoint
            saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
            saver.restore(sess, checkpoint_path)
            
            # Get all variables from the checkpoint
            tf1_vars = tf.compat.v1.global_variables()
            
            for var in tf1_vars:
                try:
                    tf1_var_values[var.name] = sess.run(var)
                except Exception as e:
                    print(f"Warning: Could not load variable {var.name}: {e}")
        
        return tf1_var_values
    
    def map_tf1_to_pytorch_weights(self, pytorch_model: nn.Module, tf1_weights: Dict[str, np.ndarray], 
                                  instrument_idx: int, instrument_name: str):
        """
        Map TF1.x variable names to PyTorch model weights for a specific instrument.
        
        Args:
            pytorch_model: PyTorch model
            tf1_weights: Dictionary of TF1 weights {var_name: value}
            instrument_idx: Index of the instrument in the original model
            instrument_name: Name of the instrument
        """
        print(f"Mapping TF1.x weights to PyTorch model for instrument: {instrument_name}")
        
        # Encoder Conv2D layers mapping
        encoder_conv_mapping = []
        for i in range(6):
            if instrument_idx == 0:
                if i == 0:
                    conv_name = 'conv2d'
                else:
                    conv_name = f'conv2d_{i}'
            else:
                conv_name = f'conv2d_{i + 7}'
            encoder_conv_mapping.append((f'{conv_name}/kernel:0', f'{conv_name}/bias:0'))
        
        # Decoder Conv2DTranspose layers mapping
        decoder_conv_mapping = []
        for i in range(6):
            if instrument_idx == 0:
                if i == 0:
                    conv_name = 'conv2d_transpose'
                else:
                    conv_name = f'conv2d_transpose_{i}'
            else:
                conv_name = f'conv2d_transpose_{i + 6}'
            decoder_conv_mapping.append((f'{conv_name}/kernel:0', f'{conv_name}/bias:0'))
        
        # Final output mask layer
        if instrument_idx == 0:
            output_conv_mapping = ('conv2d_6/kernel:0', 'conv2d_6/bias:0')
        else:
            output_conv_mapping = ('conv2d_13/kernel:0', 'conv2d_13/bias:0')
        
        # Map encoder layers
        encoder_layers = [
            pytorch_model.encoder_conv1, pytorch_model.encoder_conv2, pytorch_model.encoder_conv3,
            pytorch_model.encoder_conv4, pytorch_model.encoder_conv5, pytorch_model.encoder_conv6
        ]
        
        for i, layer in enumerate(encoder_layers):
            if i < len(encoder_conv_mapping):
                kernel_name, bias_name = encoder_conv_mapping[i]
                if kernel_name in tf1_weights and bias_name in tf1_weights:
                    # Convert TensorFlow weights (H, W, C_in, C_out) to PyTorch (C_out, C_in, H, W)
                    tf_weight = tf1_weights[kernel_name]
                    pytorch_weight = np.transpose(tf_weight, (3, 2, 0, 1))
                    
                    layer.weight.data = torch.from_numpy(pytorch_weight).float()
                    layer.bias.data = torch.from_numpy(tf1_weights[bias_name]).float()
                    print(f"  Mapped {kernel_name} to encoder_conv{i+1}")
                else:
                    print(f"  Warning: Could not find weights for encoder_conv{i+1}: {kernel_name}, {bias_name}")
        
        # Map decoder layers
        decoder_layers = [
            pytorch_model.decoder_conv1, pytorch_model.decoder_conv2, pytorch_model.decoder_conv3,
            pytorch_model.decoder_conv4, pytorch_model.decoder_conv5, pytorch_model.decoder_conv6
        ]
        
        for i, layer in enumerate(decoder_layers):
            if i < len(decoder_conv_mapping):
                kernel_name, bias_name = decoder_conv_mapping[i]
                if kernel_name in tf1_weights and bias_name in tf1_weights:
                    # Convert TensorFlow transpose conv weights (H, W, C_out, C_in) to PyTorch (C_in, C_out, H, W)
                    tf_weight = tf1_weights[kernel_name]
                    # TF Conv2DTranspose: (H, W, C_out, C_in) -> PyTorch ConvTranspose2d: (C_in, C_out, H, W)
                    # Note: TF transpose conv has C_out and C_in swapped compared to regular conv
                    pytorch_weight = np.transpose(tf_weight, (3, 2, 0, 1))  # (C_in, C_out, H, W)
                    
                    layer.weight.data = torch.from_numpy(pytorch_weight).float()
                    layer.bias.data = torch.from_numpy(tf1_weights[bias_name]).float()
                    print(f"  Mapped {kernel_name} to decoder_conv{i+1}")
                else:
                    print(f"  Warning: Could not find weights for decoder_conv{i+1}: {kernel_name}, {bias_name}")
        
        # Map output mask layer
        kernel_name, bias_name = output_conv_mapping
        if kernel_name in tf1_weights and bias_name in tf1_weights:
            tf_weight = tf1_weights[kernel_name]
            pytorch_weight = np.transpose(tf_weight, (3, 2, 0, 1))
            
            pytorch_model.mask_conv.weight.data = torch.from_numpy(pytorch_weight).float()
            pytorch_model.mask_conv.bias.data = torch.from_numpy(tf1_weights[bias_name]).float()
            print(f"  Mapped {kernel_name} to mask_conv")
        else:
            print(f"  Warning: Could not find weights for mask layer: {kernel_name}, {bias_name}")
        
        # Map batch normalization layers
        self._map_batch_norm_weights(pytorch_model, tf1_weights, instrument_idx)
    
    def _map_batch_norm_weights(self, pytorch_model: nn.Module, tf1_weights: Dict[str, np.ndarray], 
                               instrument_idx: int):
        """Map batch normalization weights."""
        
        # Get all BatchNorm layers
        bn_layers = [
            pytorch_model.encoder_bn1, pytorch_model.encoder_bn2, pytorch_model.encoder_bn3,
            pytorch_model.encoder_bn4, pytorch_model.encoder_bn5,
            pytorch_model.decoder_bn1, pytorch_model.decoder_bn2, pytorch_model.decoder_bn3,
            pytorch_model.decoder_bn4, pytorch_model.decoder_bn5, pytorch_model.decoder_bn6
        ]
        
        # TF1 BN mapping (same as Keras version)
        tf1_bn_mapping = []
        if instrument_idx == 0:
            # Encoder BN layers (5 layers: 0-4)
            tf1_bn_mapping.extend([
                'batch_normalization',      # encoder_bn1 (16)
                'batch_normalization_1',    # encoder_bn2 (32) 
                'batch_normalization_2',    # encoder_bn3 (64)
                'batch_normalization_3',    # encoder_bn4 (128)
                'batch_normalization_4',    # encoder_bn5 (256)
            ])
            # Decoder BN layers (6 layers: 6-11, skipping 5)
            tf1_bn_mapping.extend([
                'batch_normalization_6',    # decoder_bn1 (256)
                'batch_normalization_7',    # decoder_bn2 (128)
                'batch_normalization_8',    # decoder_bn3 (64)
                'batch_normalization_9',    # decoder_bn4 (32)
                'batch_normalization_10',   # decoder_bn5 (16)
                'batch_normalization_11',   # decoder_bn6 (1)
            ])
        else:
            # For accompaniment (instrument_idx=1), BN layers start from 12
            tf1_bn_mapping.extend([
                'batch_normalization_12',   # encoder_bn1 (16)
                'batch_normalization_13',   # encoder_bn2 (32)
                'batch_normalization_14',   # encoder_bn3 (64)
                'batch_normalization_15',   # encoder_bn4 (128)
                'batch_normalization_16',   # encoder_bn5 (256)
            ])
            # Decoder BN layers (skip 17, use 18-23)
            tf1_bn_mapping.extend([
                'batch_normalization_18',   # decoder_bn1 (256)
                'batch_normalization_19',   # decoder_bn2 (128)
                'batch_normalization_20',   # decoder_bn3 (64)
                'batch_normalization_21',   # decoder_bn4 (32)
                'batch_normalization_22',   # decoder_bn5 (16)
                'batch_normalization_23',   # decoder_bn6 (1)
            ])
        
        for i, layer in enumerate(bn_layers):
            if i < len(tf1_bn_mapping):
                bn_name = tf1_bn_mapping[i]
                
                gamma_name = f'{bn_name}/gamma:0'
                beta_name = f'{bn_name}/beta:0'
                mean_name = f'{bn_name}/moving_mean:0'
                var_name = f'{bn_name}/moving_variance:0'
                
                if all(name in tf1_weights for name in [gamma_name, beta_name, mean_name, var_name]):
                    layer.weight.data = torch.from_numpy(tf1_weights[gamma_name]).float()
                    layer.bias.data = torch.from_numpy(tf1_weights[beta_name]).float()
                    layer.running_mean.data = torch.from_numpy(tf1_weights[mean_name]).float()
                    layer.running_var.data = torch.from_numpy(tf1_weights[var_name]).float()
                    print(f"  Mapped BN weights for {bn_name}")
                else:
                    print(f"  Warning: Could not find BN weights for {bn_name}")
            else:
                print(f"  Warning: No TF1 mapping for BN layer {i}")
    
    def convert_model(self, model_type: str, pretrained_path: str, output_path: str):
        """
        Convert a TF1.x model to PyTorch format.
        
        Args:
            model_type: '2stems' or '4stems'
            pretrained_path: Path to pretrained model directory
            output_path: Path to save converted PyTorch model
        """
        print(f"Converting {model_type} model...")
        
        # Load configuration
        config_path = f"configs/{model_type}/base_config.json"
        config = self.load_config(config_path)
        
        # Extract instruments list
        instruments = config["instrument_list"]
        print(f"Instruments: {instruments}")
        
        # Load TF1.x weights once
        checkpoint_path = os.path.join(pretrained_path, "model")
        tf1_weights = self.load_tf1_weights_detailed(checkpoint_path)
        
        # Create individual models for each instrument
        pytorch_models = {}
        for idx, instrument in enumerate(instruments):
            print(f"\nBuilding PyTorch model for {instrument}...")
            
            # Build PyTorch model for this instrument
            pytorch_model = UNetSingleInstrument(instrument)
            print(f"Built PyTorch model for {instrument}")
            
            # Map weights for this instrument
            self.map_tf1_to_pytorch_weights(pytorch_model, tf1_weights, idx, instrument)
            
            pytorch_models[instrument] = pytorch_model
        
        # Save the converted models
        os.makedirs(output_path, exist_ok=True)
        
        for instrument, model in pytorch_models.items():
            model_path = os.path.join(output_path, f"{instrument}_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved {instrument} model to {model_path}")
        
        # Save configuration for the new models
        pytorch_config = {
            "model_type": model_type,
            "instruments": instruments,
            "sample_rate": config["sample_rate"],
            "frame_length": config["frame_length"],
            "frame_step": config["frame_step"],
            "T": config["T"],
            "F": config["F"],
            "n_channels": config["n_channels"]
        }
        
        with open(os.path.join(output_path, "config.json"), 'w') as f:
            json.dump(pytorch_config, f, indent=2)
        
        print(f"Models converted and saved to {output_path}")
        
        return pytorch_models

def main():
    """Main conversion function."""
    converter = SpleeterPyTorchConverter()
    
    # Convert 2stems model
    print("="*50)
    print("Converting 2stems model to PyTorch...")
    print("="*50)
    try:
        converter.convert_model(
            model_type="2stems",
            pretrained_path="pretrained_models/2stems",
            output_path="converted_models_pytorch/2stems"
        )
        print("✓ 2stems PyTorch model conversion completed successfully")
    except Exception as e:
        print(f"✗ Error converting 2stems model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 