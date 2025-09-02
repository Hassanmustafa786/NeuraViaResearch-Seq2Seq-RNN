import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from tqdm import tqdm
import numpy as np

from data_utils import create_dataloaders
from model import Seq2SeqRNN, count_parameters

def train_model(model, train_loader, val_loader, dataset, num_epochs=5, learning_rate=0.001, device='cpu'):
    """Train the seq2seq model"""
    
    # Setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (input_seq, target_seq, _, _) in enumerate(train_pbar):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(input_seq, target_seq, teacher_forcing_ratio=0.7)
            
            # Calculate loss
            # Reshape for loss calculation
            output = output.reshape(-1, output.size(-1))  # (batch_size * seq_length, vocab_size)
            target = target_seq[:, 1:].reshape(-1)  # Skip SOS token, flatten
            
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_sequences = 0
        total_sequences = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            
            for input_seq, target_seq, _, _ in val_pbar:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                
                # Forward pass
                output = model(input_seq, target_seq, teacher_forcing_ratio=0.0)  # No teacher forcing in validation
                
                # Calculate loss
                output_loss = output.reshape(-1, output.size(-1))
                target_loss = target_seq[:, 1:].reshape(-1)
                loss = criterion(output_loss, target_loss)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Calculate accuracy (exact sequence match)
                predicted = output.argmax(dim=-1)
                target_no_sos = target_seq[:, 1:]  # Remove SOS token
                
                for i in range(predicted.size(0)):
                    pred_seq = predicted[i]
                    true_seq = target_no_sos[i]
                    
                    # Find EOS positions
                    pred_eos = (pred_seq == 2).nonzero(as_tuple=True)[0]
                    true_eos = (true_seq == 2).nonzero(as_tuple=True)[0]
                    
                    pred_len = pred_eos[0].item() if len(pred_eos) > 0 else len(pred_seq)
                    true_len = true_eos[0].item() if len(true_eos) > 0 else len(true_seq)
                    
                    # Compare sequences up to EOS
                    if pred_len == true_len and torch.equal(pred_seq[:pred_len], true_seq[:true_len]):
                        correct_sequences += 1
                    
                    total_sequences += 1
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_batches
        val_accuracy = correct_sequences / total_sequences
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Show some predictions
        if epoch % 2 == 0:  # Every 2 epochs
            show_predictions(model, val_loader, dataset, device, num_examples=3)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def show_predictions(model, val_loader, dataset, device, num_examples=3):
    """Show some example predictions"""
    model.eval()
    with torch.no_grad():
        examples_shown = 0
        for input_seq, target_seq, digit_seqs, word_seqs in val_loader:
            if examples_shown >= num_examples:
                break
            
            input_seq = input_seq.to(device)
            
            for i in range(min(len(digit_seqs), num_examples - examples_shown)):
                # Get prediction
                pred_indices = model.predict(input_seq[i])
                predicted_words = dataset.decode_output(pred_indices)
                
                print(f"  Input: '{digit_seqs[i]}' -> True: '{word_seqs[i]}' | Pred: '{predicted_words}'")
                
                examples_shown += 1
                if examples_shown >= num_examples:
                    break
    print()

def save_model(model, dataset, training_history, save_dir='models'):
    """Save the trained model and metadata"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_dir, 'seq2seq_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save model architecture and vocabulary info
    model_info = {
        'input_vocab_size': len(dataset.input_vocab),
        'output_vocab_size': len(dataset.output_vocab),
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'input_vocab': dataset.input_vocab,
        'output_vocab': dataset.output_vocab,
        'max_input_length': dataset.max_input_length,
        'max_output_length': dataset.max_output_length,
        'training_history': training_history
    }
    
    info_path = os.path.join(save_dir, 'model_info.pth')
    torch.save(model_info, info_path)
    
    print(f"Model saved to {model_path}")
    print(f"Model info saved to {info_path}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data
    train_loader, val_loader, dataset = create_dataloaders(
        num_samples=1000,
        batch_size=32,
        train_ratio=0.8
    )
    
    # Get dataset reference for vocabulary info
    if hasattr(train_loader.dataset, 'dataset'):
        data_ref = train_loader.dataset.dataset
    else:
        data_ref = train_loader.dataset
    
    # Create model
    model = Seq2SeqRNN(
        input_vocab_size=len(data_ref.input_vocab),
        output_vocab_size=len(data_ref.output_vocab),
        hidden_size=128,
        num_layers=2
    )
    
    print("Starting training...")
    
    # Train model
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=data_ref,
        num_epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    # Save model
    save_model(model, data_ref, training_history)
    
    print("\nTraining completed!")
    print(f"Final validation accuracy: {training_history['val_accuracies'][-1]:.4f}")

if __name__ == "__main__":
    main()
