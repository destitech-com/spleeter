#!/usr/bin/env python3
"""
Improved script to convert Spleeter TensorFlow 1.x checkpoint models to modern Keras format.

This script loads the pretrained TF 1.x models and converts them to Keras models
that can be used with modern TensorFlow versions.
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Disable TF2 behavior to work with TF1 checkpoints
tf.compat.v1.disable_v2_behavior()

class SpleeterKerasConverter:
    """Converts Spleeter TF1.x models to Keras format."""
    
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
    
    def build_keras_unet_single(self, instrument_name: str, config: Dict[str, Any]) -> tf.keras.Model:
        """
        Build a single Keras U-Net model for one instrument.
        
        Args:
            instrument_name: Name of the instrument
            config: Model configuration dictionary
            
        Returns:
            Keras model for single instrument
        """
        from tensorflow.keras.layers import (
            Input, Conv2D, Conv2DTranspose, BatchNormalization, 
            Dropout, Concatenate, Multiply, ELU, ReLU, LeakyReLU
        )
        from tensorflow.keras.initializers import HeUniform
        from tensorflow.keras.models import Model
        
        # Input layer - spectrogram input
        input_shape = (None, self.F, self.n_channels)  # (T, F, channels)
        inputs = Input(shape=input_shape, name='input_spectrogram')
        
        # Encoder path - exactly matching original
        conv_n_filters = [16, 32, 64, 128, 256, 512]
        kernel_initializer = HeUniform(seed=50)
        conv_activation_layer = LeakyReLU(0.2)
        deconv_activation_layer = ReLU()
        
        # Encoder layers - exactly matching original order
        conv1 = Conv2D(conv_n_filters[0], (5, 5), strides=(2, 2), padding='same', 
                      kernel_initializer=kernel_initializer, name='encoder_conv1')(inputs)
        batch1 = BatchNormalization(axis=-1, name='encoder_bn1')(conv1)
        rel1 = conv_activation_layer(batch1)
        
        conv2 = Conv2D(conv_n_filters[1], (5, 5), strides=(2, 2), padding='same',
                      kernel_initializer=kernel_initializer, name='encoder_conv2')(rel1)
        batch2 = BatchNormalization(axis=-1, name='encoder_bn2')(conv2)
        rel2 = conv_activation_layer(batch2)
        
        conv3 = Conv2D(conv_n_filters[2], (5, 5), strides=(2, 2), padding='same',
                      kernel_initializer=kernel_initializer, name='encoder_conv3')(rel2)
        batch3 = BatchNormalization(axis=-1, name='encoder_bn3')(conv3)
        rel3 = conv_activation_layer(batch3)
        
        conv4 = Conv2D(conv_n_filters[3], (5, 5), strides=(2, 2), padding='same',
                      kernel_initializer=kernel_initializer, name='encoder_conv4')(rel3)
        batch4 = BatchNormalization(axis=-1, name='encoder_bn4')(conv4)
        rel4 = conv_activation_layer(batch4)
        
        conv5 = Conv2D(conv_n_filters[4], (5, 5), strides=(2, 2), padding='same',
                      kernel_initializer=kernel_initializer, name='encoder_conv5')(rel4)
        batch5 = BatchNormalization(axis=-1, name='encoder_bn5')(conv5)
        rel5 = conv_activation_layer(batch5)
        
        conv6 = Conv2D(conv_n_filters[5], (5, 5), strides=(2, 2), padding='same',
                      kernel_initializer=kernel_initializer, name='encoder_conv6')(rel5)
        # Note: batch6 and rel6 are not used in the original, so we skip them
        
        # Decoder path - exactly matching original order and connections
        up1 = Conv2DTranspose(conv_n_filters[4], (5, 5), strides=(2, 2), padding='same',
                             kernel_initializer=kernel_initializer, name='decoder_conv1')(conv6)  # Use conv6, not rel6!
        up1 = deconv_activation_layer(up1)
        batch7 = BatchNormalization(axis=-1, name='decoder_bn1')(up1)
        drop1 = Dropout(0.5)(batch7)
        merge1 = Concatenate(axis=-1, name='skip_conn1')([conv5, drop1])  # Use conv5, not rel5!
        
        up2 = Conv2DTranspose(conv_n_filters[3], (5, 5), strides=(2, 2), padding='same',
                             kernel_initializer=kernel_initializer, name='decoder_conv2')(merge1)
        up2 = deconv_activation_layer(up2)
        batch8 = BatchNormalization(axis=-1, name='decoder_bn2')(up2)
        drop2 = Dropout(0.5)(batch8)
        merge2 = Concatenate(axis=-1, name='skip_conn2')([conv4, drop2])  # Use conv4, not rel4!
        
        up3 = Conv2DTranspose(conv_n_filters[2], (5, 5), strides=(2, 2), padding='same',
                             kernel_initializer=kernel_initializer, name='decoder_conv3')(merge2)
        up3 = deconv_activation_layer(up3)
        batch9 = BatchNormalization(axis=-1, name='decoder_bn3')(up3)
        drop3 = Dropout(0.5)(batch9)
        merge3 = Concatenate(axis=-1, name='skip_conn3')([conv3, drop3])  # Use conv3, not rel3!
        
        up4 = Conv2DTranspose(conv_n_filters[1], (5, 5), strides=(2, 2), padding='same',
                             kernel_initializer=kernel_initializer, name='decoder_conv4')(merge3)
        up4 = deconv_activation_layer(up4)
        batch10 = BatchNormalization(axis=-1, name='decoder_bn4')(up4)
        merge4 = Concatenate(axis=-1, name='skip_conn4')([conv2, batch10])  # Use conv2, not rel2!
        
        up5 = Conv2DTranspose(conv_n_filters[0], (5, 5), strides=(2, 2), padding='same',
                             kernel_initializer=kernel_initializer, name='decoder_conv5')(merge4)
        up5 = deconv_activation_layer(up5)
        batch11 = BatchNormalization(axis=-1, name='decoder_bn5')(up5)
        merge5 = Concatenate(axis=-1, name='skip_conn5')([conv1, batch11])  # Use conv1, not rel1!
        
        up6 = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                             kernel_initializer=kernel_initializer, name='decoder_conv6')(merge5)
        up6 = deconv_activation_layer(up6)
        batch12 = BatchNormalization(axis=-1, name='decoder_bn6')(up6)
        
        # Output layer - exactly matching original
        mask = Conv2D(2, (4, 4), dilation_rate=(2, 2), activation='sigmoid',
                     padding='same', kernel_initializer=kernel_initializer,
                     name=f'{instrument_name}_mask')(batch12)
        output = Multiply(name=f'{instrument_name}_output')([mask, inputs])
        
        # Create model
        model = Model(inputs=inputs, outputs=output, name=f'spleeter_{instrument_name}')
        
        return model
    
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
    
    def map_tf1_to_keras_weights(self, keras_model: tf.keras.Model, tf1_weights: Dict[str, np.ndarray], 
                                instrument_idx: int, instrument_name: str):
        """
        Map TF1.x variable names to Keras layer weights for a specific instrument.
        
        Args:
            keras_model: Keras model
            tf1_weights: Dictionary of TF1 weights {var_name: value}
            instrument_idx: Index of the instrument in the original model (0 for first instrument, 1 for second, etc.)
            instrument_name: Name of the instrument
        """
        print(f"Mapping TF1.x weights to Keras model for instrument: {instrument_name}")
        
        # For 2-stems: vocals (idx=0) uses conv2d, conv2d_1 to conv2d_5, accompaniment (idx=1) uses conv2d_7 to conv2d_12
        # Encoder Conv2D layers (6 layers)
        encoder_conv_mapping = []
        for i in range(6):
            if instrument_idx == 0:
                if i == 0:
                    conv_name = 'conv2d'  # First layer has no number
                else:
                    conv_name = f'conv2d_{i}'
            else:
                conv_name = f'conv2d_{i + 7}'
            encoder_conv_mapping.append((f'{conv_name}/kernel:0', f'{conv_name}/bias:0'))
        
        # Decoder Conv2DTranspose layers (6 layers)
        decoder_conv_mapping = []
        for i in range(6):
            if instrument_idx == 0:
                if i == 0:
                    conv_name = 'conv2d_transpose'  # First transpose layer has no number
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
        encoder_layers = [l for l in keras_model.layers if 'encoder_conv' in l.name]
        for i, layer in enumerate(encoder_layers):
            if i < len(encoder_conv_mapping):
                kernel_name, bias_name = encoder_conv_mapping[i]
                if kernel_name in tf1_weights and bias_name in tf1_weights:
                    layer.set_weights([tf1_weights[kernel_name], tf1_weights[bias_name]])
                    print(f"  Mapped {kernel_name} to {layer.name}")
                else:
                    print(f"  Warning: Could not find weights for {layer.name}: {kernel_name}, {bias_name}")
        
        # Map decoder layers
        decoder_layers = [l for l in keras_model.layers if 'decoder_conv' in l.name]
        for i, layer in enumerate(decoder_layers):
            if i < len(decoder_conv_mapping):
                kernel_name, bias_name = decoder_conv_mapping[i]
                if kernel_name in tf1_weights and bias_name in tf1_weights:
                    layer.set_weights([tf1_weights[kernel_name], tf1_weights[bias_name]])
                    print(f"  Mapped {kernel_name} to {layer.name}")
                else:
                    print(f"  Warning: Could not find weights for {layer.name}: {kernel_name}, {bias_name}")
        
        # Map output mask layer
        mask_layers = [l for l in keras_model.layers if f'{instrument_name}_mask' in l.name]
        if mask_layers:
            layer = mask_layers[0]
            kernel_name, bias_name = output_conv_mapping
            if kernel_name in tf1_weights and bias_name in tf1_weights:
                layer.set_weights([tf1_weights[kernel_name], tf1_weights[bias_name]])
                print(f"  Mapped {kernel_name} to {layer.name}")
            else:
                print(f"  Warning: Could not find weights for mask layer: {kernel_name}, {bias_name}")
        
        # Map batch normalization layers
        self._map_batch_norm_weights(keras_model, tf1_weights, instrument_idx)
    
    def _map_batch_norm_weights(self, keras_model: tf.keras.Model, tf1_weights: Dict[str, np.ndarray], 
                               instrument_idx: int):
        """Map batch normalization weights."""
        bn_layers = [l for l in keras_model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
        
        # Expected TF1 BN names for vocals (instrument_idx=0):
        # Encoder: batch_normalization (16), batch_normalization_1 (32), batch_normalization_2 (64), 
        #          batch_normalization_3 (128), batch_normalization_4 (256)
        # Skip: batch_normalization_5 (512) - not used in our model
        # Decoder: batch_normalization_6 (256), batch_normalization_7 (128), batch_normalization_8 (64),
        #          batch_normalization_9 (32), batch_normalization_10 (16), batch_normalization_11 (1)
        
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
            # Encoder BN layers
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
                    layer.set_weights([
                        tf1_weights[gamma_name],
                        tf1_weights[beta_name],
                        tf1_weights[mean_name],
                        tf1_weights[var_name]
                    ])
                    print(f"  Mapped BN weights for {layer.name} ({bn_name})")
                else:
                    print(f"  Warning: Could not find BN weights for {layer.name}: {bn_name}")
            else:
                print(f"  Warning: No TF1 mapping for BN layer {layer.name}")
    
    def convert_model(self, model_type: str, pretrained_path: str, output_path: str):
        """
        Convert a TF1.x model to Keras format.
        
        Args:
            model_type: '2stems' or '4stems'
            pretrained_path: Path to pretrained model directory
            output_path: Path to save converted Keras model
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
        keras_models = {}
        for idx, instrument in enumerate(instruments):
            print(f"\nBuilding model for {instrument}...")
            
            # Build Keras model for this instrument
            keras_model = self.build_keras_unet_single(instrument, config)
            print(f"Built Keras model for {instrument}")
            
            # Map weights for this instrument
            self.map_tf1_to_keras_weights(keras_model, tf1_weights, idx, instrument)
            
            keras_models[instrument] = keras_model
        
        # Save the converted models
        os.makedirs(output_path, exist_ok=True)
        
        for instrument, model in keras_models.items():
            model_path = os.path.join(output_path, f"{instrument}_model.keras")
            model.save(model_path)
            print(f"Saved {instrument} model to {model_path}")
        
        # Save configuration for the new models
        keras_config = {
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
            json.dump(keras_config, f, indent=2)
        
        print(f"Models converted and saved to {output_path}")
        
        return keras_models

def main():
    """Main conversion function."""
    converter = SpleeterKerasConverter()
    
    # Convert 2stems model
    print("="*50)
    print("Converting 2stems model...")
    print("="*50)
    try:
        converter.convert_model(
            model_type="2stems",
            pretrained_path="pretrained_models/2stems",
            output_path="converted_models/2stems"
        )
        print("✓ 2stems model conversion completed successfully")
    except Exception as e:
        print(f"✗ Error converting 2stems model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 