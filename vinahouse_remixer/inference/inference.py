"""
Inference pipeline for Vinahouse Remixer.

This module implements the inference pipeline for transforming
songs into Vinahouse style using the trained diffusion model.
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, List, Optional, Union
import argparse
import json
from tqdm import tqdm

from vinahouse_remixer.preprocessing.audio_processor import AudioPreprocessor
from vinahouse_remixer.style_transfer.diffusion import DiffusionModel, StyleTransferModule
from vinahouse_remixer.rhythm_transform.rhythm_transformer import RhythmTransformer
from vinahouse_remixer.remixer import VinahouseRemixer


def mel_to_audio(mel_spectrogram: np.ndarray, 
                sample_rate: int = 44100,
                n_fft: int = 2048,
                hop_length: int = 512) -> np.ndarray:
    """
    Convert mel-spectrogram back to audio using Griffin-Lim algorithm.
    
    Args:
        mel_spectrogram: Mel-spectrogram as numpy array
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        Reconstructed audio as numpy array
    """
    # Convert from dB scale
    mel_spec = librosa.db_to_power(mel_spectrogram)
    
    # Convert mel-spectrogram to magnitude spectrogram
    mag_spec = librosa.feature.inverse.mel_to_stft(
        mel_spec, 
        sr=sample_rate, 
        n_fft=n_fft
    )
    
    # Reconstruct phase and convert to audio
    audio = librosa.griffinlim(
        mag_spec, 
        hop_length=hop_length,
        n_iter=32  # More iterations for better quality
    )
    
    return audio


def process_file(file_path: str, 
                output_dir: str,
                model_path: str,
                config_path: str,
                style_strength: float = 0.8) -> str:
    """
    Process a single audio file to transform it to Vinahouse style.
    
    Args:
        file_path: Path to input audio file
        output_dir: Directory to save output
        model_path: Path to trained model weights
        config_path: Path to configuration file
        style_strength: Strength of style transfer (0.0 to 1.0)
        
    Returns:
        Path to output file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file path
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    output_file = os.path.join(output_dir, f"{name}_vinahouse{ext}")
    
    # Initialize remixer
    remixer = VinahouseRemixer(
        model_path=model_path,
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        style_strength=style_strength
    )
    
    # Remix the file
    output_file = remixer.remix(file_path, output_file)
    
    return output_file


def batch_process(input_dir: str, 
                 output_dir: str,
                 model_path: str,
                 config_path: str,
                 style_strength: float = 0.8,
                 file_ext: str = '.wav') -> List[str]:
    """
    Process all audio files in a directory to transform them to Vinahouse style.
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save output
        model_path: Path to trained model weights
        config_path: Path to configuration file
        style_strength: Strength of style transfer (0.0 to 1.0)
        file_ext: File extension to process
        
    Returns:
        List of paths to output files
    """
    # Get list of input files
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                  if f.endswith(file_ext)]
    
    if not input_files:
        print(f"No {file_ext} files found in {input_dir}")
        return []
    
    # Initialize remixer
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    remixer = VinahouseRemixer(
        model_path=model_path,
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        style_strength=style_strength
    )
    
    # Process files
    return remixer.batch_remix(input_files, output_dir)


def main():
    """
    Main function for the inference script.
    """
    parser = argparse.ArgumentParser(description="Transform songs to Vinahouse style")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input audio file or directory")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output file or directory")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to trained model weights")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to configuration file")
    parser.add_argument("--style-strength", type=float, default=0.8, 
                        help="Strength of style transfer (0.0 to 1.0)")
    parser.add_argument("--batch", action="store_true", 
                        help="Process all files in input directory")
    parser.add_argument("--file-ext", type=str, default=".wav", 
                        help="File extension to process in batch mode")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing
        output_files = batch_process(
            args.input, 
            args.output, 
            args.model, 
            args.config, 
            args.style_strength,
            args.file_ext
        )
        print(f"Processed {len(output_files)} files:")
        for f in output_files:
            print(f"  {f}")
    else:
        # Single file processing
        output_file = process_file(
            args.input, 
            args.output, 
            args.model, 
            args.config, 
            args.style_strength
        )
        print(f"Processed file saved to: {output_file}")


if __name__ == "__main__":
    main()
