import torch
import os
import numpy as np
from model import Seq2SeqRNN
from data_utils import DigitToWordDataset

def load_model(model_dir='models'):
    """Load the trained model and its metadata"""
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    model_path = os.path.join(model_dir, 'seq2seq_model.pth')
    
    if not os.path.exists(model_info_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    # Load model info
    model_info = torch.load(model_info_path, map_location='cpu')
    
    # Create model with saved architecture
    model = Seq2SeqRNN(
        input_vocab_size=model_info['input_vocab_size'],
        output_vocab_size=model_info['output_vocab_size'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers']
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, model_info

def create_test_dataset(model_info):
    """Create a dataset with the same configuration as training"""
    dataset = DigitToWordDataset(num_samples=100)  # Smaller test set
    
    # Ensure vocabularies match
    dataset.input_vocab = model_info['input_vocab']
    dataset.output_vocab = model_info['output_vocab']
    dataset.input_idx_to_token = {v: k for k, v in dataset.input_vocab.items()}
    dataset.output_idx_to_token = {v: k for k, v in dataset.output_vocab.items()}
    dataset.max_input_length = model_info['max_input_length']
    dataset.max_output_length = model_info['max_output_length']
    
    return dataset

def test_single_prediction(model, dataset, digit_sequence):
    """Test model on a single digit sequence"""
    print(f"\nTesting: '{digit_sequence}'")
    
    # Encode input
    input_tensor = dataset.encode_input(digit_sequence)
    
    # Get prediction
    with torch.no_grad():
        predicted_indices = model.predict(input_tensor)
        predicted_words = dataset.decode_output(predicted_indices)
    
    # Get ground truth
    digit_to_word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    true_words = ' '.join([digit_to_word[digit] for digit in digit_sequence])
    
    print(f"True:      '{true_words}'")
    print(f"Predicted: '{predicted_words}'")
    
    # Check if prediction is correct
    is_correct = predicted_words == true_words
    print(f"Correct:   {is_correct}")
    
    return is_correct, predicted_words, true_words

def evaluate_model(model, dataset, num_test_samples=50):
    """Evaluate model on multiple test samples"""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    correct_predictions = 0
    total_predictions = 0
    test_results = []
    
    # Generate test cases
    test_cases = []
    
    # Single digits
    for i in range(10):
        test_cases.append(str(i))
    
    # Two digits
    for _ in range(15):
        test_cases.append(''.join([str(np.random.randint(0, 10)) for _ in range(2)]))
    
    # Three digits
    for _ in range(15):
        test_cases.append(''.join([str(np.random.randint(0, 10)) for _ in range(3)]))
    
    # Four digits
    for _ in range(10):
        test_cases.append(''.join([str(np.random.randint(0, 10)) for _ in range(4)]))
    
    print(f"Testing on {len(test_cases)} samples...")
    print("-" * 60)
    
    for i, digit_seq in enumerate(test_cases):
        is_correct, predicted, true = test_single_prediction(model, dataset, digit_seq)
        
        test_results.append({
            'input': digit_seq,
            'true': true,
            'predicted': predicted,
            'correct': is_correct
        })
        
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        if i < 10:  # Show first 10 detailed results
            print()
    
    # Calculate metrics
    accuracy = correct_predictions / total_predictions
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Test Cases: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Analyze by sequence length
    length_stats = {}
    for result in test_results:
        length = len(result['input'])
        if length not in length_stats:
            length_stats[length] = {'correct': 0, 'total': 0}
        length_stats[length]['total'] += 1
        if result['correct']:
            length_stats[length]['correct'] += 1
    
    print("\nAccuracy by sequence length:")
    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        acc = stats['correct'] / stats['total']
        print(f"  Length {length}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    # Show some error cases
    error_cases = [r for r in test_results if not r['correct']]
    if error_cases:
        print(f"\nError Analysis (showing up to 5 errors):")
        for i, error in enumerate(error_cases[:5]):
            print(f"  {i+1}. Input: '{error['input']}' | True: '{error['true']}' | Predicted: '{error['predicted']}'")
    
    return {
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'length_stats': length_stats,
        'test_results': test_results
    }

def interactive_test(model, dataset):
    """Interactive testing mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING MODE")
    print("=" * 60)
    print("Enter digit sequences to test the model.")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nEnter digit sequence: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Validate input
            if not user_input.isdigit():
                print("Please enter only digits (0-9).")
                continue
            
            if len(user_input) == 0:
                print("Please enter at least one digit.")
                continue
            
            if len(user_input) > 5:
                print("Please enter at most 5 digits.")
                continue
            
            # Test the input
            test_single_prediction(model, dataset, user_input)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main testing function"""
    print("Loading trained model...")
    
    try:
        model, model_info = load_model()
        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create test dataset
        dataset = create_test_dataset(model_info)
        
        # Run evaluation
        evaluation_results = evaluate_model(model, dataset)
        
        # Save evaluation results
        os.makedirs('results', exist_ok=True)
        torch.save(evaluation_results, 'results/test_results.pth')
        print(f"\nTest results saved to 'results/test_results.pth'")
        
        # Interactive testing
        interactive_test(model, dataset)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to train the model.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
