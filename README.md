# Seq2Seq RNN for Digit-to-Word Conversion

This project implements a complete sequence-to-sequence RNN pipeline that converts digit sequences to their word equivalents. For example, "235" becomes "two three five".

## Project Structure

```
NeuraViaResearch/
├── data_utils.py      # Data generation and preprocessing utilities
├── model.py           # Seq2Seq RNN model architecture
├── train.py          # Training script (50 epochs)
├── test.py           # Model testing and evaluation
├── visualization.py  # Performance metrics and visualizations
├── requirements.txt  # Python dependencies
├── models/           # Directory for saved models
├── visualizations/   # Directory for generated plots and metrics
└── results/          # Directory for test results
```

## Features

- **Synthetic Data Generation**: Creates balanced datasets with equal representation of digits 0-9
- **Seq2Seq Architecture**: Encoder-decoder LSTM model with attention-like mechanisms
- **Comprehensive Training**: 5-epoch training with validation monitoring
- **Model Export**: Automatic model saving to `models/` directory
- **Interactive Testing**: Real-time testing interface
- **Rich Visualizations**: Training curves, accuracy analysis, confusion matrices
- **Performance Metrics**: Detailed evaluation reports

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
This will:
- Generate 1000 synthetic training samples
- Train for 5 epochs
- Save the model to `models/`
- Show training progress and sample predictions

### 3. Test the Model
```bash
python test.py
```
This will:
- Load the trained model
- Evaluate on test cases of varying lengths
- Provide accuracy metrics
- Offer interactive testing mode

### 4. Generate Visualizations
```bash
python visualization.py
```
This will create:
- Training curves plot
- Accuracy by sequence length analysis
- Confusion matrix for digit-level predictions
- Comprehensive performance summary
- Detailed metrics report

## Model Architecture

The seq2seq model consists of:

- **Encoder**: LSTM that processes input digit sequences
- **Decoder**: LSTM that generates output word sequences
- **Vocabulary**: 
  - Input: digits 0-9 + special tokens (PAD, SOS, EOS)
  - Output: words "zero" through "nine" + special tokens
- **Training Features**:
  - Teacher forcing during training
  - Gradient clipping to prevent exploding gradients
  - Cross-entropy loss with padding token ignored

## Example Usage

### Training Output
```
Training on device: cpu
Model parameters: 85,389
Training samples: 800
Validation samples: 200

Epoch 1/5
Train Loss: 2.1543 | Val Loss: 1.8765 | Val Acc: 0.2150
  Input: '42' -> True: 'four two' | Pred: 'four two'
  Input: '8' -> True: 'eight' | Pred: 'eight'

...
```

### Test Results
```
Testing: '235'
True:      'two three five'
Predicted: 'two three five'
Correct:   True

EVALUATION RESULTS
Total Test Cases: 50
Correct Predictions: 47
Accuracy: 0.9400 (94.00%)
```

## Performance Expectations

With the small-scale setup:
- **Single digits**: ~95-100% accuracy
- **Two digits**: ~85-95% accuracy  
- **Three digits**: ~80-90% accuracy
- **Four+ digits**: ~70-85% accuracy

The model typically achieves good performance on this constrained task due to:
- Simple vocabulary (only 10 digits/words)
- Regular patterns in the data
- Sufficient training samples for the task complexity

## Customization

You can modify various parameters:

- **Dataset size**: Change `num_samples` in `create_dataloaders()`
- **Model architecture**: Adjust `hidden_size` and `num_layers` in `Seq2SeqRNN`
- **Training duration**: Modify `num_epochs` in `train_model()`
- **Sequence lengths**: Update `min_length` and `max_length` in `DigitToWordDataset`

## Files Description

- **`data_utils.py`**: Handles synthetic data generation, vocabulary creation, and data loading
- **`model.py`**: Contains the seq2seq model architecture with encoder-decoder LSTMs
- **`train.py`**: Complete training pipeline with progress monitoring and model saving
- **`test.py`**: Comprehensive testing with evaluation metrics and interactive mode
- **`visualization.py`**: Creates detailed performance visualizations and reports

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, Matplotlib, Seaborn, Pandas, tqdm

The project is designed to run efficiently on CPU and works well for educational and research purposes.
