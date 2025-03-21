"""
Audio preprocessing module for Vinahouse Remixer.

This module handles audio loading, segmentation, feature extraction,
and mel-spectrogram conversion.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Tuple, List, Optional, Union


class AudioPreprocessor:
    """
    Handles preprocessing of audio files for Vinahouse remixing.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 80,
                 segment_duration: float = 5.0):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            n_fft: FFT window size for spectrogram computation
            hop_length: Hop length for spectrogram computation
            n_mels: Number of mel bands to generate
            segment_duration: Duration of audio segments in seconds
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_duration = segment_duration
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono if necessary.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio, sr
    
    def segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Segment audio into fixed-length chunks.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            List of audio segments
        """
        segment_length = int(self.sample_rate * self.segment_duration)
        segments = []
        
        # Calculate number of segments
        num_segments = int(np.ceil(len(audio) / segment_length))
        
        for i in range(num_segments):
            start = i * segment_length
            end = min(start + segment_length, len(audio))
            segment = audio[start:end]
            
            # Pad last segment if needed
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))
            
            segments.append(segment)
        
        return segments
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Extract audio features useful for remixing.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features['tempo'] = tempo
        
        # Extract beats
        _, beat_frames = librosa.beat.beat_track(y=audio, sr=self.sample_rate, 
                                                hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate, 
                                           hop_length=self.hop_length)
        features['beat_times'] = beat_times
        
        # Extract harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic'] = harmonic
        features['percussive'] = percussive
        
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid'] = spectral_centroid
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms'] = rms
        
        return features
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel-spectrogram from audio data.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Mel-spectrogram as numpy array
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Convert to power spectrogram
        power_spec = np.abs(stft) ** 2
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            S=power_spec, 
            sr=self.sample_rate, 
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def process_file(self, file_path: str) -> Dict[str, any]:
        """
        Process an audio file: load, segment, extract features, and compute mel-spectrograms.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing processed data
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Extract global features
        features = self.extract_features(audio)
        
        # Segment audio
        segments = self.segment_audio(audio)
        
        # Compute mel-spectrograms for each segment
        mel_specs = [self.compute_mel_spectrogram(segment) for segment in segments]
        
        return {
            'file_path': file_path,
            'audio': audio,
            'sample_rate': sr,
            'features': features,
            'segments': segments,
            'mel_spectrograms': mel_specs
        }
