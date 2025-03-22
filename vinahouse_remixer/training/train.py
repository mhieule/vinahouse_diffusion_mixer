"""
Training pipeline for Vinahouse Remixer.

This module implements the training pipeline for the diffusion model
used in the Vinahouse remixer.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import glob
from typing import Dict, Tuple, List, Optional, Union
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from vinahouse_remixer.preprocessing.audio_processor import AudioPreprocessor
from vinahouse_remixer.style_transfer.diffusion import DiffusionModel


class VinahouseDataset(Dataset):
    """
    Dataset for training the Vinahouse remixer.
    """
    
    def __init__(self, 
                 original_dir: str,
                 vinahouse_dir: str,
                 preprocessor: Optional[AudioPreprocessor] = None,
                 paired: bool = False,
                 max_samples: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            original_dir: Directory containing original songs
            vinahouse_dir: Directory containing Vinahouse songs
            preprocessor: Audio preprocessor instance
            paired: Whether the data is paired (original and Vinahouse versions of same songs)
            max_samples: Maximum number of samples to load
        """
        self.original_dir = original_dir
        self.vinahouse_dir = vinahouse_dir
        self.paired = paired
        
        # Initialize preprocessor if not provided
        self.preprocessor = preprocessor or AudioPreprocessor()
        
        # Get file lists
        self.original_files = sorted(glob.glob(os.path.join(original_dir, "*.mp3")))
        self.vinahouse_files = sorted(glob.glob(os.path.join(vinahouse_dir, "*.mp3")))
        print("DEBUG: Original directory:", original_dir)
        print("DEBUG: Vinahouse directory:", vinahouse_dir)
        print("DEBUG: Original files (up to 5):", self.original_files[:5])
        print("DEBUG: Vinahouse files (up to 5):", self.vinahouse_files[:5])
        
        # Limit number of samples if specified
        if max_samples is not None:
            self.original_files = self.original_files[:max_samples]
            self.vinahouse_files = self.vinahouse_files[:max_samples]
        
        # Ensure equal number of files for paired data
        if paired:
            min_files = min(len(self.original_files), len(self.vinahouse_files))
            self.original_files = self.original_files[:min_files]
            self.vinahouse_files = self.vinahouse_files[:min_files]
        
        print(f"Loaded {len(self.original_files)} original files and {len(self.vinahouse_files)} Vinahouse files")
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Number of samples in the dataset
        """
        if self.paired:
            return len(self.original_files)
        else:
            return len(self.original_files) + len(self.vinahouse_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing mel-spectrograms
        """
        if self.paired:
            # For paired data, return both original and Vinahouse versions
            original_file = self.original_files[idx]
            vinahouse_file = self.vinahouse_files[idx]
            
            # Process files
            original_data = self.preprocessor.process_file(original_file)
            vinahouse_data = self.preprocessor.process_file(vinahouse_file)
            
            # Get first segment for simplicity
            original_mel = original_data['mel_spectrograms'][0]
            vinahouse_mel = vinahouse_data['mel_spectrograms'][0]
            
            return {
                'original_mel': torch.from_numpy(original_mel).float(),
                'vinahouse_mel': torch.from_numpy(vinahouse_mel).float()
            }
        else:
            # For unpaired data, return either original or Vinahouse
            if idx < len(self.original_files):
                # Original song
                file_path = self.original_files[idx]
                is_vinahouse = False
            else:
                # Vinahouse song
                file_path = self.vinahouse_files[idx - len(self.original_files)]
                is_vinahouse = True
            
            # Process file
            data = self.preprocessor.process_file(file_path)
            
            # Get first segment for simplicity
            mel = data['mel_spectrograms'][0]
            
            return {
                'mel': torch.from_numpy(mel).float(),
                'is_vinahouse': torch.tensor(is_vinahouse).float()
            }


class DiffusionTrainer:
    """
    Trainer for the diffusion model.
    """
    
    def __init__(self, 
                 model: DiffusionModel,
                 train_dataset: VinahouseDataset,
                 val_dataset: Optional[VinahouseDataset] = None,
                 batch_size: int = 16,
                 learning_rate: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the trainer.
        
        Args:
            model: Diffusion model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        print("DEBUG: Total samples in train dataset:", len(train_dataset))
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4
            )
        else:
            self.val_loader = None
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize loss function
        self.loss_fn = nn.MSELoss()
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Process batch based on dataset type
            if 'original_mel' in batch:
                # Paired data
                original_mel = batch['original_mel'].unsqueeze(1).to(self.device)
                vinahouse_mel = batch['vinahouse_mel'].unsqueeze(1).to(self.device)
                
                # Forward pass
                noise = torch.randn_like(original_mel)
                t = torch.randint(0, self.model.time_steps, (original_mel.shape[0],), device=self.device)
                
                # Add noise to original mel-spectrogram
                noised_mel = self.model.q_sample(original_mel, t, noise)
                
                # Predict noise
                predicted_noise = self.model.predict_noise(noised_mel, t)
                
                # Compute loss
                loss = self.loss_fn(predicted_noise, noise)
            else:
                # Unpaired data
                mel = batch['mel'].unsqueeze(1).to(self.device)
                is_vinahouse = batch['is_vinahouse'].to(self.device)
                
                # Forward pass
                noise = torch.randn_like(mel)
                t = torch.randint(0, self.model.time_steps, (mel.shape[0],), device=self.device)
                
                # Add noise to mel-spectrogram
                noised_mel = self.model.q_sample(mel, t, noise)
                
                # Predict noise
                predicted_noise = self.model.predict_noise(noised_mel, t)
                
                # Compute loss
                loss = self.loss_fn(predicted_noise, noise)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Process batch based on dataset type
                if 'original_mel' in batch:
                    # Paired data
                    original_mel = batch['original_mel'].unsqueeze(1).to(self.device)
                    vinahouse_mel = batch['vinahouse_mel'].unsqueeze(1).to(self.device)
                    
                    # Forward pass
                    noise = torch.randn_like(original_mel)
                    t = torch.randint(0, self.model.time_steps, (original_mel.shape[0],), device=self.device)
                    
                    # Add noise to original mel-spectrogram
                    noised_mel = self.model.q_sample(original_mel, t, noise)
                    
                    # Predict noise
                    predicted_noise = self.model.predict_noise(noised_mel, t)
                    
                    # Compute loss
                    loss = self.loss_fn(predicted_noise, noise)
                else:
                    # Unpaired data
                    mel = batch['mel'].unsqueeze(1).to(self.device)
                    is_vinahouse = batch['is_vinahouse'].to(self.device)
                    
                    # Forward pass
                    noise = torch.randn_like(mel)
                    t = torch.randint(0, self.model.time_steps, (mel.shape[0],), device=self.device)
                    
                    # Add noise to mel-spectrogram
                    noised_mel = self.model.q_sample(mel, t, noise)
                    
                    # Predict noise
                    predicted_noise = self.model.predict_noise(noised_mel, t)
                    
                    # Compute loss
                    loss = self.loss_fn(predicted_noise, noise)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self, 
              num_epochs: int, 
              checkpoint_dir: str,
              save_every: int = 5) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            print(f"Train loss: {train_loss:.6f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                print(f"Validation loss: {val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Save training history
                history_path = os.path.join(checkpoint_dir, "training_history.json")
                with open(history_path, 'w') as f:
                    json.dump(self.history, f)
        
        # Save final model
        final_path = os.path.join(checkpoint_dir, "model_final.pt")
        torch.save(self.model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")
        
        # Plot training history
        self._plot_history(os.path.join(checkpoint_dir, "training_history.png"))
        
        return self.history
    
    def _plot_history(self, save_path: str):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        
        if self.val_loader is not None:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()


def train_model(config_path: str):
    """
    Train the diffusion model using configuration from a JSON file.
    
    Args:
        config_path: Path to configuration JSON file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=config.get('sample_rate', 44100),
        n_fft=config.get('n_fft', 2048),
        hop_length=config.get('hop_length', 512),
        n_mels=config.get('n_mels', 80),
        segment_duration=config.get('segment_duration', 5.0)
    )
    
    # Create datasets
    train_dataset = VinahouseDataset(
        original_dir=config['original_train_dir'],
        vinahouse_dir=config['vinahouse_train_dir'],
        preprocessor=preprocessor,
        paired=config.get('paired_data', False),
        max_samples=config.get('max_train_samples')
    )
    
    if 'original_val_dir' in config and 'vinahouse_val_dir' in config:
        val_dataset = VinahouseDataset(
            original_dir=config['original_val_dir'],
            vinahouse_dir=config['vinahouse_val_dir'],
            preprocessor=preprocessor,
            paired=config.get('paired_data', False),
            max_samples=config.get('max_val_samples')
        )
    else:
        val_dataset = None
    
    # Create model
    model = DiffusionModel(
        latent_dim=config.get('latent_dim', 64),
        time_steps=config.get('time_steps', 1000),
        beta_schedule=config.get('beta_schedule', 'linear'),
        beta_start=config.get('beta_start', 1e-4),
        beta_end=config.get('beta_end', 2e-2)
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.get('batch_size', 16),
        learning_rate=config.get('learning_rate', 1e-4),
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Train model
    trainer.train(
        num_epochs=config.get('num_epochs', 100),
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        save_every=config.get('save_every', 5)
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Vinahouse remixer diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    
    args = parser.parse_args()
    train_model(args.config)
