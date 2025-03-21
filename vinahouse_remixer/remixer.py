"""
Main remixer class for Vinahouse Remixer.

This module integrates the preprocessing, style transfer, and rhythm transformation
components to provide a complete pipeline for remixing songs to Vinahouse style.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, List, Optional, Union

from vinahouse_remixer.preprocessing.audio_processor import AudioPreprocessor
from vinahouse_remixer.style_transfer.diffusion import StyleTransferModule
from vinahouse_remixer.rhythm_transform.rhythm_transformer import RhythmTransformer


class VinahouseRemixer:
    """
    Main class for remixing songs to Vinahouse style.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 style_strength: float = 0.8):
        """
        Initialize the Vinahouse remixer.
        
        Args:
            model_path: Path to pretrained diffusion model weights
            device: Device to run the model on (cuda or cpu)
            style_strength: Strength of style transfer (0.0 to 1.0)
        """
        self.style_strength = style_strength
        
        # Initialize components
        self.preprocessor = AudioPreprocessor()
        self.style_transfer = StyleTransferModule(model_path=model_path, device=device)
        self.rhythm_transformer = RhythmTransformer()
    
    def remix(self, input_file: str, output_file: str) -> str:
        """
        Remix a song to Vinahouse style.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to save remixed audio
            
        Returns:
            Path to remixed audio file
        """
        print(f"Remixing {input_file} to Vinahouse style...")
        
        # Step 1: Preprocess audio
        print("Preprocessing audio...")
        processed_data = self.preprocessor.process_file(input_file)
        
        # Step 2: Apply style transfer to mel-spectrograms
        print("Applying spectral style transfer...")
        transformed_mels = self.style_transfer.batch_transform(
            processed_data['mel_spectrograms'], 
            style_strength=self.style_strength
        )
        
        # Step 3: Apply rhythm and tempo transformation
        print("Transforming rhythm and tempo...")
        transformed_audio = self.rhythm_transformer.transform(
            processed_data['audio'],
            processed_data['features']
        )
        
        # Step 4: Save remixed audio
        print(f"Saving remixed audio to {output_file}...")
        sf.write(output_file, transformed_audio, processed_data['sample_rate'])
        
        return output_file
    
    def batch_remix(self, input_files: List[str], output_dir: str) -> List[str]:
        """
        Remix multiple songs to Vinahouse style.
        
        Args:
            input_files: List of paths to input audio files
            output_dir: Directory to save remixed audio files
            
        Returns:
            List of paths to remixed audio files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        
        for input_file in input_files:
            # Generate output file path
            filename = os.path.basename(input_file)
            name, ext = os.path.splitext(filename)
            output_file = os.path.join(output_dir, f"{name}_vinahouse{ext}")
            
            # Remix the file
            remixed_file = self.remix(input_file, output_file)
            output_files.append(remixed_file)
        
        return output_files
