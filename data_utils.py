import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class DigitToWordDataset(Dataset):
    """Dataset for converting digit sequences to words"""
    
    def __init__(self, num_samples=1000, min_length=1, max_length=5):
        self.digit_to_word = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        # Generate synthetic data
        self.data = self._generate_data(num_samples, min_length, max_length)
        
        # Create vocabularies
        self._create_vocabularies()
        
    def _generate_data(self, num_samples, min_length, max_length):
        """Generate synthetic digit sequences and their word equivalents"""
        data = []
        
        # Ensure equal distribution of each digit
        for _ in range(num_samples):
            # Random length between min_length and max_length
            length = random.randint(min_length, max_length)
            
            # Generate digit sequence
            digits = [str(random.randint(0, 9)) for _ in range(length)]
            digit_sequence = ''.join(digits)
            
            # Convert to words
            words = [self.digit_to_word[digit] for digit in digits]
            word_sequence = ' '.join(words)
            
            data.append((digit_sequence, word_sequence))
        
        return data
    
    def _create_vocabularies(self):
        """Create input and output vocabularies"""
        # Input vocabulary (digits)
        self.input_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        for i in range(10):
            self.input_vocab[str(i)] = len(self.input_vocab)
        
        # Output vocabulary (words)
        self.output_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        for word in words:
            self.output_vocab[word] = len(self.output_vocab)
        
        # Reverse vocabularies
        self.input_idx_to_token = {v: k for k, v in self.input_vocab.items()}
        self.output_idx_to_token = {v: k for k, v in self.output_vocab.items()}
        
        # Calculate max lengths
        self.max_input_length = max(len(x[0]) for x in self.data) + 2  # +2 for SOS/EOS
        self.max_output_length = max(len(x[1].split()) for x in self.data) + 2  # +2 for SOS/EOS
    
    def encode_input(self, digit_sequence):
        """Encode digit sequence to tensor"""
        tokens = ['<SOS>'] + list(digit_sequence) + ['<EOS>']
        indices = [self.input_vocab[token] for token in tokens]
        
        # Pad to max length
        while len(indices) < self.max_input_length:
            indices.append(self.input_vocab['<PAD>'])
            
        return torch.tensor(indices, dtype=torch.long)
    
    def encode_output(self, word_sequence):
        """Encode word sequence to tensor"""
        tokens = ['<SOS>'] + word_sequence.split() + ['<EOS>']
        indices = [self.output_vocab[token] for token in tokens]
        
        # Pad to max length
        while len(indices) < self.max_output_length:
            indices.append(self.output_vocab['<PAD>'])
            
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_output(self, indices):
        """Decode output indices to words"""
        tokens = []
        for idx in indices:
            token = self.output_idx_to_token[idx.item()]
            if token == '<EOS>':
                break
            if token not in ['<PAD>', '<SOS>']:
                tokens.append(token)
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        digit_seq, word_seq = self.data[idx]
        input_tensor = self.encode_input(digit_seq)
        target_tensor = self.encode_output(word_seq)
        
        return input_tensor, target_tensor, digit_seq, word_seq

def create_dataloaders(num_samples=1000, batch_size=32, train_ratio=0.8):
    """Create train and validation dataloaders"""
    # Create dataset
    dataset = DigitToWordDataset(num_samples=num_samples)
    
    # Split into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset

if __name__ == "__main__":
    # Test the dataset
    dataset = DigitToWordDataset(num_samples=20)
    print("Sample data:")
    for i in range(5):
        input_tensor, target_tensor, digit_seq, word_seq = dataset[i]
        print(f"Input: {digit_seq} -> Output: {word_seq}")
        print(f"Encoded input: {input_tensor}")
        print(f"Encoded target: {target_tensor}")
        print()
