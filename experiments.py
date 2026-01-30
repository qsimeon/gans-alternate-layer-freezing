"""Experiments comparing standard vs. alternating training approaches.

This module runs experiments to compare:
1. Standard training (update all parameters)
2. Alternating training (freeze one half while updating the other)
3. Batch-level alternating training

It evaluates both training speed and final performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import SimpleNN
from training_loop import train_standard, train_alternating, train_alternating_batch, evaluate_model


def load_mnist_data(test_size=0.2, batch_size=128):
    """Load and preprocess MNIST dataset.
    
    Args:
        test_size (float): Fraction of data to use for testing
        batch_size (int): Batch size for DataLoaders
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    print('Loading MNIST dataset...')
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist['data'], mnist['target']
    
    # Convert to numpy arrays
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    # Convert to float32 and normalize
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def plot_results(results_dict, save_path='comparison_results.png'):
    """Plot comparison of different training approaches.
    
    Args:
        results_dict (dict): Dictionary containing results for each approach
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, results in results_dict.items():
        ax.plot(results['losses'], label=name, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax = axes[0, 1]
    for name, results in results_dict.items():
        ax.plot(results['accuracies'], label=name, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Time
    ax = axes[1, 0]
    names = list(results_dict.keys())
    times = [results_dict[name]['time'] for name in names]
    bars = ax.bar(names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Total Training Time Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    # Plot 4: Final Accuracy
    ax = axes[1, 1]
    final_accs = [results_dict[name]['accuracies'][-1] for name in names]
    bars = ax.bar(names, final_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Final Test Accuracy Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([min(final_accs) - 5, 100])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.show()


def print_summary(results_dict):
    """Print summary statistics for all approaches.
    
    Args:
        results_dict (dict): Dictionary containing results for each approach
    """
    print('\n' + '='*70)
    print('SUMMARY OF RESULTS')
    print('='*70)
    
    for name, results in results_dict.items():
        print(f'\n{name}:')
        print(f"  Final Test Accuracy: {results['accuracies'][-1]:.2f}%")
        print(f"  Final Training Loss: {results['losses'][-1]:.4f}")
        print(f"  Total Training Time: {results['time']:.2f} seconds")
        print(f"  Time per Epoch: {results['time']/len(results['losses']):.2f} seconds")
    
    print('\n' + '='*70)
    print('COMPARISON')
    print('='*70)
    
    # Compare to standard approach
    standard_acc = results_dict['Standard']['accuracies'][-1]
    standard_time = results_dict['Standard']['time']
    
    for name, results in results_dict.items():
        if name == 'Standard':
            continue
        
        acc_diff = results['accuracies'][-1] - standard_acc
        time_diff = results['time'] - standard_time
        time_ratio = results['time'] / standard_time
        
        print(f'\n{name} vs Standard:')
        print(f"  Accuracy Difference: {acc_diff:+.2f}%")
        print(f"  Time Difference: {time_diff:+.2f} seconds ({time_ratio:.2f}x)")
    
    print('\n' + '='*70)


def run_experiments(num_epochs=15, learning_rate=0.001, batch_size=128, device='cpu'):
    """Run all experiments and compare results.
    
    Args:
        num_epochs (int): Number of epochs to train each model
        learning_rate (float): Learning rate for optimization
        batch_size (int): Batch size for training
        device (str): Device to run experiments on ('cpu' or 'cuda')
    """
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=batch_size)
    
    results = {}
    
    # Experiment 1: Standard Training
    print('\n' + '='*70)
    print('EXPERIMENT 1: STANDARD TRAINING')
    print('='*70)
    model_standard = SimpleNN()
    losses_std, accs_std, time_std = train_standard(
        model_standard, train_loader, test_loader,
        num_epochs=num_epochs, learning_rate=learning_rate, device=device
    )
    results['Standard'] = {
        'losses': losses_std,
        'accuracies': accs_std,
        'time': time_std
    }
    
    # Experiment 2: Alternating Training (epoch-level)
    print('\n' + '='*70)
    print('EXPERIMENT 2: ALTERNATING TRAINING (EPOCH-LEVEL)')
    print('='*70)
    model_alternating = SimpleNN()
    losses_alt, accs_alt, time_alt = train_alternating(
        model_alternating, train_loader, test_loader,
        num_epochs=num_epochs, learning_rate=learning_rate, device=device,
        switch_frequency=1
    )
    results['Alternating (Epoch)'] = {
        'losses': losses_alt,
        'accuracies': accs_alt,
        'time': time_alt
    }
    
    # Experiment 3: Alternating Training (batch-level)
    print('\n' + '='*70)
    print('EXPERIMENT 3: ALTERNATING TRAINING (BATCH-LEVEL)')
    print('='*70)
    model_alternating_batch = SimpleNN()
    losses_alt_batch, accs_alt_batch, time_alt_batch = train_alternating_batch(
        model_alternating_batch, train_loader, test_loader,
        num_epochs=num_epochs, learning_rate=learning_rate, device=device
    )
    results['Alternating (Batch)'] = {
        'losses': losses_alt_batch,
        'accuracies': accs_alt_batch,
        'time': time_alt_batch
    }
    
    # Print summary
    print_summary(results)
    
    # Plot results
    plot_results(results)
    
    return results


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Run experiments
    results = run_experiments(
        num_epochs=15,
        learning_rate=0.001,
        batch_size=128,
        device=device
    )
    
    print('\nExperiments completed successfully!')
