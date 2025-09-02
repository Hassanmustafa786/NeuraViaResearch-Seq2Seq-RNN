import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
import pandas as pd

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_training_history(model_dir='models'):
    """Load training history from saved model info"""
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    
    if not os.path.exists(model_info_path):
        raise FileNotFoundError("Model info file not found. Please train the model first.")
    
    model_info = torch.load(model_info_path, map_location='cpu')
    return model_info['training_history']

def load_test_results(results_dir='results'):
    """Load test results"""
    test_results_path = os.path.join(results_dir, 'test_results.pth')
    
    if not os.path.exists(test_results_path):
        raise FileNotFoundError("Test results file not found. Please run test.py first.")
    
    return torch.load(test_results_path, map_location='cpu')

def plot_training_curves(training_history, save_dir='visualizations'):
    """Plot training and validation loss curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(training_history['train_losses']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, training_history['val_accuracies'], 'g-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    
    plt.show()
    return fig

def plot_accuracy_by_length(test_results, save_dir='visualizations'):
    """Plot accuracy by sequence length"""
    os.makedirs(save_dir, exist_ok=True)
    
    length_stats = test_results['length_stats']
    
    lengths = sorted(length_stats.keys())
    accuracies = [length_stats[length]['correct'] / length_stats[length]['total'] for length in lengths]
    counts = [length_stats[length]['total'] for length in lengths]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy by length
    bars1 = ax1.bar(lengths, accuracies, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Sequence Length')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Sample count by length
    bars2 = ax2.bar(lengths, counts, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Number of Test Cases')
    ax2.set_title('Test Cases by Sequence Length')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'accuracy_by_length.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy by length plot saved to {save_path}")
    
    plt.show()
    return fig

def plot_confusion_matrix(test_results, save_dir='visualizations'):
    """Create a confusion matrix for digit predictions"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract digit-level predictions
    all_true_digits = []
    all_pred_digits = []
    
    for result in test_results['test_results']:
        true_words = result['true'].split()
        pred_words = result['predicted'].split()
        
        # Convert words back to digits
        word_to_digit = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        
        # Only consider cases where prediction length matches true length
        if len(true_words) == len(pred_words):
            for true_word, pred_word in zip(true_words, pred_words):
                if true_word in word_to_digit and pred_word in word_to_digit:
                    all_true_digits.append(word_to_digit[true_word])
                    all_pred_digits.append(word_to_digit[pred_word])
    
    # Create confusion matrix
    digits = [str(i) for i in range(10)]
    confusion_matrix = np.zeros((10, 10))
    
    for true_digit, pred_digit in zip(all_true_digits, all_pred_digits):
        true_idx = int(true_digit)
        pred_idx = int(pred_digit)
        confusion_matrix[true_idx, pred_idx] += 1
    
    # Normalize by row (true labels)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(confusion_matrix, row_sums, 
                             out=np.zeros_like(confusion_matrix), 
                             where=row_sums!=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(normalized_cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=digits, yticklabels=digits, ax=ax)
    ax.set_xlabel('Predicted Digit')
    ax.set_ylabel('True Digit')
    ax.set_title('Digit-Level Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return fig

def create_performance_summary(training_history, test_results, save_dir='visualizations'):
    """Create a comprehensive performance summary"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate metrics
    final_train_loss = training_history['train_losses'][-1]
    final_val_loss = training_history['val_losses'][-1]
    final_val_accuracy = training_history['val_accuracies'][-1]
    test_accuracy = test_results['accuracy']
    
    # Create summary figure
    fig = plt.figure(figsize=(16, 10))
    
    # Define grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Training curves
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(training_history['train_losses']) + 1)
    ax1.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy progression
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, training_history['val_accuracies'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Accuracy by length
    ax3 = fig.add_subplot(gs[1, :2])
    length_stats = test_results['length_stats']
    lengths = sorted(length_stats.keys())
    accuracies = [length_stats[length]['correct'] / length_stats[length]['total'] for length in lengths]
    ax3.bar(lengths, accuracies, color='lightblue', alpha=0.7, edgecolor='navy')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Test Accuracy by Length')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Performance metrics text
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    metrics_text = f"""
    PERFORMANCE SUMMARY
    
    Training:
    • Final Train Loss: {final_train_loss:.4f}
    • Final Val Loss: {final_val_loss:.4f}
    • Final Val Accuracy: {final_val_accuracy:.4f}
    
    Testing:
    • Test Accuracy: {test_accuracy:.4f}
    • Total Test Cases: {test_results['total_predictions']}
    • Correct Predictions: {test_results['correct_predictions']}
    
    Model Performance:
    • Single digits: {length_stats.get(1, {}).get('correct', 0)}/{length_stats.get(1, {}).get('total', 0)}
    • Two digits: {length_stats.get(2, {}).get('correct', 0)}/{length_stats.get(2, {}).get('total', 0)}
    • Three digits: {length_stats.get(3, {}).get('correct', 0)}/{length_stats.get(3, {}).get('total', 0)}
    """
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Error analysis
    ax5 = fig.add_subplot(gs[2, :])
    error_cases = [r for r in test_results['test_results'] if not r['correct']]
    
    if error_cases:
        # Sample some errors for display
        sample_errors = error_cases[:8]  # Show up to 8 errors
        error_text = "SAMPLE ERRORS:\n\n"
        for i, error in enumerate(sample_errors):
            error_text += f"{i+1}. '{error['input']}' → True: '{error['true']}' | Pred: '{error['predicted']}'\n"
    else:
        error_text = "NO ERRORS - PERFECT PERFORMANCE!"
    
    ax5.axis('off')
    ax5.text(0.02, 0.98, error_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    ax5.set_title('Error Analysis', y=0.95)
    
    plt.suptitle('Seq2Seq RNN Performance Report', fontsize=16, fontweight='bold')
    
    # Save plot
    save_path = os.path.join(save_dir, 'performance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance summary saved to {save_path}")
    
    plt.show()
    return fig

def save_metrics_report(training_history, test_results, save_dir='visualizations'):
    """Save detailed metrics to a text file"""
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, 'metrics_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SEQ2SEQ RNN PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Training metrics
        f.write("TRAINING METRICS:\n")
        f.write("-"*20 + "\n")
        f.write(f"Number of epochs: {len(training_history['train_losses'])}\n")
        f.write(f"Final training loss: {training_history['train_losses'][-1]:.6f}\n")
        f.write(f"Final validation loss: {training_history['val_losses'][-1]:.6f}\n")
        f.write(f"Final validation accuracy: {training_history['val_accuracies'][-1]:.6f}\n")
        f.write(f"Best validation accuracy: {max(training_history['val_accuracies']):.6f}\n\n")
        
        # Test metrics
        f.write("TEST METRICS:\n")
        f.write("-"*15 + "\n")
        f.write(f"Test accuracy: {test_results['accuracy']:.6f}\n")
        f.write(f"Correct predictions: {test_results['correct_predictions']}\n")
        f.write(f"Total predictions: {test_results['total_predictions']}\n\n")
        
        # Length-wise performance
        f.write("PERFORMANCE BY SEQUENCE LENGTH:\n")
        f.write("-"*35 + "\n")
        length_stats = test_results['length_stats']
        for length in sorted(length_stats.keys()):
            stats = length_stats[length]
            accuracy = stats['correct'] / stats['total']
            f.write(f"Length {length}: {accuracy:.4f} ({stats['correct']}/{stats['total']})\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Detailed metrics report saved to {report_path}")

def main():
    """Main visualization function"""
    print("Loading data for visualization...")
    
    try:
        # Load training history and test results
        training_history = load_training_history()
        test_results = load_test_results()
        
        print("Creating visualizations...")
        
        # Create all visualizations
        plot_training_curves(training_history)
        plot_accuracy_by_length(test_results)
        plot_confusion_matrix(test_results)
        create_performance_summary(training_history, test_results)
        
        # Save detailed report
        save_metrics_report(training_history, test_results)
        
        print("\nAll visualizations completed and saved to 'visualizations/' directory!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run train.py and test.py first.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main()
