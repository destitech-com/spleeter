#!/usr/bin/env python3
"""
Step-by-step comparison between Keras and PyTorch models to identify divergence.
"""

import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from convert_models_to_pytorch import UNetSingleInstrument
tf.compat.v1.enable_eager_execution() 

def load_pytorch_model(model_path: str, instrument: str, device: str = 'cpu'):
    """Load a PyTorch model."""
    model = UNetSingleInstrument(instrument)
    model_file = os.path.join(model_path, f"{instrument}_model.pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    model.to(device)
    return model

def load_keras_model(model_path: str, instrument: str):
    """Load a Keras model."""
    model_file = os.path.join(model_path, f"{instrument}_model.keras")
    return tf.keras.models.load_model(model_file)

def manual_keras_forward(model, input_tensor):
    """Manually run Keras model step by step."""
    print("\n=== KERAS STEP-BY-STEP ===")
    
    # Get layers by name
    layers = {layer.name: layer for layer in model.layers}
    
    x = input_tensor
    print(f"Input: {x.shape} ({x.numpy().min():.6f}, {x.numpy().max():.6f})")
    
    # Encoder path
    conv1 = layers['encoder_conv1'](x)
    batch1 = layers['encoder_bn1'](conv1)
    rel1 = tf.nn.leaky_relu(batch1, alpha=0.2)
    print(f"Step1 (conv1): {conv1.shape} ({conv1.numpy().min():.6f}, {conv1.numpy().max():.6f})")
    print(f"Step1 (after bn): { batch1.shape} ({batch1.numpy().min():.6f}, {batch1.numpy().max():.6f})")
    print(f"Step1 (after bn+relu): {rel1.shape} ({rel1.numpy().min():.6f}, {rel1.numpy().max():.6f})")
    
    conv2 = layers['encoder_conv2'](rel1)
    batch2 = layers['encoder_bn2'](conv2)
    rel2 = tf.nn.leaky_relu(batch2, alpha=0.2)
    print(f"Step2 (conv2): {conv2.shape} ({conv2.numpy().min():.6f}, {conv2.numpy().max():.6f})")
    print(f"Step2 (after bn+relu): {rel2.shape} ({rel2.numpy().min():.6f}, {rel2.numpy().max():.6f})")
    
    conv3 = layers['encoder_conv3'](rel2)
    batch3 = layers['encoder_bn3'](conv3)
    rel3 = tf.nn.leaky_relu(batch3, alpha=0.2)
    print(f"Step3 (conv3): {conv3.shape} ({conv3.numpy().min():.6f}, {conv3.numpy().max():.6f})")
    print(f"Step3 (after bn+relu): {rel3.shape} ({rel3.numpy().min():.6f}, {rel3.numpy().max():.6f})")
    
    conv4 = layers['encoder_conv4'](rel3)
    batch4 = layers['encoder_bn4'](conv4)
    rel4 = tf.nn.leaky_relu(batch4, alpha=0.2)
    print(f"Step4 (conv4): {conv4.shape} ({conv4.numpy().min():.6f}, {conv4.numpy().max():.6f})")
    print(f"Step4 (after bn+relu): {rel4.shape} ({rel4.numpy().min():.6f}, {rel4.numpy().max():.6f})")
    
    conv5 = layers['encoder_conv5'](rel4)
    batch5 = layers['encoder_bn5'](conv5)
    rel5 = tf.nn.leaky_relu(batch5, alpha=0.2)
    print(f"Step5 (conv5): {conv5.shape} ({conv5.numpy().min():.6f}, {conv5.numpy().max():.6f})")
    print(f"Step5 (after bn+relu): {rel5.shape} ({rel5.numpy().min():.6f}, {rel5.numpy().max():.6f})")
    
    conv6 = layers['encoder_conv6'](rel5)
    print(f"Step6 (conv6): {conv6.shape} ({conv6.numpy().min():.6f}, {conv6.numpy().max():.6f})")
    dropout_rate = 0.0
    # Decoder path
    up1 = layers['decoder_conv1'](conv6)
    up1 = tf.nn.relu(up1)
    batch7 = layers['decoder_bn1'](up1)
    drop1 = tf.nn.dropout(batch7, rate=dropout_rate)  # training=False for inference
    merge1 = tf.concat([conv5, drop1], axis=-1)
    print(f"Step7 (batch7): {batch7.shape} ({batch7.numpy().min():.6f}, {batch7.numpy().max():.6f})")
    print(f"Step7 (drop1): {drop1.shape} ({drop1.numpy().min():.6f}, {drop1.numpy().max():.6f})")
    print(f"Step7 (merge1): {merge1.shape} ({merge1.numpy().min():.6f}, {merge1.numpy().max():.6f})")
    
    up2 = layers['decoder_conv2'](merge1)
    up2 = tf.nn.relu(up2)
    batch8 = layers['decoder_bn2'](up2)
    drop2 = tf.nn.dropout(batch8, rate=dropout_rate)
    merge2 = tf.concat([conv4, drop2], axis=-1)
    print(f"Step8 (batch8): {batch8.shape} ({batch8.numpy().min():.6f}, {batch8.numpy().max():.6f})")
    print(f"Step8 (drop2): {drop2.shape} ({drop2.numpy().min():.6f}, {drop2.numpy().max():.6f})")
    print(f"Step8 (merge2): {merge2.shape} ({merge2.numpy().min():.6f}, {merge2.numpy().max():.6f})")
    
    up3 = layers['decoder_conv3'](merge2)
    up3 = tf.nn.relu(up3)
    batch9 = layers['decoder_bn3'](up3)
    drop3 = tf.nn.dropout(batch9, rate=dropout_rate)
    merge3 = tf.concat([conv3, drop3], axis=-1)
    print(f"Step9 (merge3): {merge3.shape} ({merge3.numpy().min():.6f}, {merge3.numpy().max():.6f})")
    
    up4 = layers['decoder_conv4'](merge3)
    up4 = tf.nn.relu(up4)
    batch10 = layers['decoder_bn4'](up4)
    merge4 = tf.concat([conv2, batch10], axis=-1)  # No dropout here
    print(f"Step10 (merge4): {merge4.shape} ({merge4.numpy().min():.6f}, {merge4.numpy().max():.6f})")
    
    up5 = layers['decoder_conv5'](merge4)
    up5 = tf.nn.relu(up5)
    batch11 = layers['decoder_bn5'](up5)
    merge5 = tf.concat([conv1, batch11], axis=-1)  # No dropout here
    print(f"Step11 (merge5): {merge5.shape} ({merge5.numpy().min():.6f}, {merge5.numpy().max():.6f})")
    
    up6 = layers['decoder_conv6'](merge5)
    up6 = tf.nn.relu(up6)
    batch12 = layers['decoder_bn6'](up6)
    print(f"Step12 (batch12): {batch12.shape} ({batch12.numpy().min():.6f}, {batch12.numpy().max():.6f})")
    
    # Find mask layer
    mask_layer = None
    for layer in model.layers:
        if 'mask' in layer.name:
            mask_layer = layer
            break
    
    if mask_layer:
        mask = mask_layer(batch12)
        print(f"Step13 (mask): {mask.shape} ({mask.numpy().min():.6f}, {mask.numpy().max():.6f})")
        
        # Apply mask to input
        output = mask * input_tensor
        print(f"Final output: {output.shape} ({output.numpy().min():.6f}, {output.numpy().max():.6f})")
        
        return output
    else:
        print("Warning: Could not find mask layer")
        return batch12

def manual_pytorch_forward(model, input_tensor):
    """Manually run PyTorch model step by step."""
    print("\n=== PYTORCH STEP-BY-STEP ===")
    
    with torch.no_grad():
        output = model.forward(input_tensor)
        print(f"Final output: {output.shape} ({output.min():.6f}, {output.max():.6f})")
        
        return output

def compare_step_by_step():
    """Compare models step by step."""
    
    # Load input tensor
    if not os.path.exists('input_tensor_spectrogram.pkl'):
        print("Error: input_tensor_spectrogram.pkl not found. Please run keras_separator.py first.")
        return
    
    with open('input_tensor_spectrogram.pkl', 'rb') as f:
        input_np = pickle.load(f)
    
    print(f"Loaded input tensor shape: {input_np.shape}")
    
    # Convert to tensors
    keras_input = tf.convert_to_tensor(input_np)
    pytorch_input = torch.from_numpy(input_np).float()
    
    # Load models
    keras_vocals = load_keras_model("converted_models/2stems", "vocals")
    pytorch_vocals = load_pytorch_model("converted_models_pytorch/2stems", "vocals", 'cpu')
    
    print("="*60)
    print("COMPARING VOCALS MODEL STEP BY STEP")
    print("="*60)
    
    # Run step by step
    keras_output = manual_keras_forward(keras_vocals, keras_input)
    pytorch_output = manual_pytorch_forward(pytorch_vocals, pytorch_input)
    
    # Final comparison
    print("\n=== FINAL COMPARISON ===")
    keras_final = keras_output.numpy()
    pytorch_final = pytorch_output.cpu().numpy()
    
    diff = np.abs(keras_final - pytorch_final)
    print(f"Final output difference: max={diff.max():.6f}, mean={diff.mean():.6f}")

if __name__ == "__main__":
    compare_step_by_step() 