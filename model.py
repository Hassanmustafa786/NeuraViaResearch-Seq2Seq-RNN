import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder part of seq2seq model"""
    
    def __init__(self, input_vocab_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    """Decoder part of seq2seq model"""
    
    def __init__(self, output_vocab_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_vocab_size = output_vocab_size
        
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_vocab_size)
        
    def forward(self, input_token, hidden_state):
        # input_token shape: (batch_size, 1)
        embedded = self.embedding(input_token)
        output, hidden_state = self.lstm(embedded, hidden_state)
        output = self.out(output)
        return output, hidden_state

class Seq2SeqRNN(nn.Module):
    """Complete Seq2Seq model for digit to word conversion"""
    
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size=128, num_layers=2):
        super(Seq2SeqRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_vocab_size = output_vocab_size
        
        self.encoder = Encoder(input_vocab_size, hidden_size, num_layers)
        self.decoder = Decoder(output_vocab_size, hidden_size, num_layers)
        
    def forward(self, input_seq, target_seq=None, max_length=20, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        
        # Encode input sequence
        encoder_outputs, encoder_hidden = self.encoder(input_seq)
        
        # Initialize decoder
        decoder_hidden = encoder_hidden
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long) * 1  # SOS token (index 1)
        
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
        
        outputs = []
        
        if target_seq is not None:  # Training mode
            target_length = target_seq.size(1)
            for t in range(target_length - 1):  # -1 because we don't predict after EOS
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs.append(decoder_output)
                
                # Teacher forcing
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    decoder_input = target_seq[:, t:t+1]  # Next target token
                else:
                    decoder_input = decoder_output.argmax(dim=-1)  # Predicted token
        else:  # Inference mode
            for t in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs.append(decoder_output)
                
                decoder_input = decoder_output.argmax(dim=-1)
                
                # Stop if all sequences in batch have produced EOS
                if (decoder_input == 2).all():  # EOS token (index 2)
                    break
        
        return torch.cat(outputs, dim=1)  # (batch_size, seq_length, vocab_size)
    
    def predict(self, input_seq, max_length=20):
        """Generate prediction for a single input sequence"""
        self.eval()
        with torch.no_grad():
            if len(input_seq.shape) == 1:
                input_seq = input_seq.unsqueeze(0)  # Add batch dimension
            
            output = self.forward(input_seq, target_seq=None, max_length=max_length)
            predicted_indices = output.argmax(dim=-1)
            
            return predicted_indices.squeeze(0)  # Remove batch dimension

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    input_vocab_size = 13  # 10 digits + PAD, SOS, EOS
    output_vocab_size = 13  # 10 words + PAD, SOS, EOS
    
    model = Seq2SeqRNN(input_vocab_size, output_vocab_size, hidden_size=64, num_layers=1)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_length = 7
    input_seq = torch.randint(3, input_vocab_size, (batch_size, seq_length))
    target_seq = torch.randint(3, output_vocab_size, (batch_size, seq_length))
    
    output = model(input_seq, target_seq)
    print(f"Input shape: {input_seq.shape}")
    print(f"Target shape: {target_seq.shape}")
    print(f"Output shape: {output.shape}")
