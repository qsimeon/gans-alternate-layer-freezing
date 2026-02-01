"""Experiments comparing standard vs. alternating training approaches.

This module runs experiments to compare:
1. Standard training (update all parameters)
2. Alternating training (freeze one half while updating the other)
3. Batch-level alternating training

It evaluates both training speed and final performance with beautiful visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from models import SimpleNN
from training_loop import train_standard, train_alternating, train_alternating_batch, evaluate_model


# Set up beautiful plot styling
def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Custom color palette - vibrant but professional
    colors = {
        'standard': '#2E86AB',      # Deep blue
        'alternating_epoch': '#A23B72',  # Magenta
        'alternating_batch': '#F18F01',  # Orange
        'background': '#1a1a2e',     # Dark background
        'grid': '#404040',           # Subtle grid
        'text': '#e0e0e0',           # Light text
    }
    
    return colors


def load_cifar10_data(batch_size=128, data_dir='./data'):
    """Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for DataLoaders
        data_dir (str): Directory to store/load dataset
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    print('📥 Loading CIFAR-10 dataset...')
    
    # CIFAR-10 normalization values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    print(f'   ✓ Train size: {len(train_dataset):,}')
    print(f'   ✓ Test size: {len(test_dataset):,}')
    print(f'   ✓ Classes: {train_dataset.classes}')
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def plot_results(results_dict, save_path='comparison_results.png'):
    """Create beautiful visualizations comparing training approaches.
    
    Args:
        results_dict (dict): Dictionary containing results for each approach
        save_path (str): Path to save the plot
    """
    colors = setup_plot_style()
    
    # Define colors for each approach
    approach_colors = {
        'Standard': colors['standard'],
        'Alternating (Epoch)': colors['alternating_epoch'],
        'Alternating (Batch)': colors['alternating_batch']
    }
    
    # Create figure with dark theme
    fig = plt.figure(figsize=(16, 12), facecolor='#0f0f1a')
    
    # Create a grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                          left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # Add main title
    fig.suptitle('GAN-Inspired Alternating Layer Freezing\nTraining Strategy Comparison on MNIST', 
                 fontsize=20, fontweight='bold', color='white', y=0.98)
    
    # ==================== PLOT 1: Training Loss Curves ====================
    ax1 = fig.add_subplot(gs[0, :2], facecolor='#1a1a2e')
    
    for name, results in results_dict.items():
        epochs = range(1, len(results['losses']) + 1)
        ax1.plot(epochs, results['losses'], 
                label=name, 
                color=approach_colors.get(name, '#ffffff'),
                linewidth=2.5,
                marker='o',
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=0.5,
                alpha=0.9)
    
    ax1.set_xlabel('Epoch', fontsize=12, color='white')
    ax1.set_ylabel('Training Loss', fontsize=12, color='white')
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold', color='white', pad=10)
    ax1.legend(fontsize=10, facecolor='#2a2a4a', edgecolor='#404080', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='#404080')
    ax1.set_facecolor('#1a1a2e')
    for spine in ax1.spines.values():
        spine.set_color('#404080')
    
    # ==================== PLOT 2: Accuracy Curves ====================
    ax2 = fig.add_subplot(gs[1, :2], facecolor='#1a1a2e')
    
    for name, results in results_dict.items():
        epochs = range(1, len(results['accuracies']) + 1)
        ax2.plot(epochs, results['accuracies'], 
                label=name,
                color=approach_colors.get(name, '#ffffff'),
                linewidth=2.5,
                marker='s',
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=0.5,
                alpha=0.9)
    
    ax2.set_xlabel('Epoch', fontsize=12, color='white')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, color='white')
    ax2.set_title('Test Accuracy Over Time', fontsize=14, fontweight='bold', color='white', pad=10)
    ax2.legend(fontsize=10, facecolor='#2a2a4a', edgecolor='#404080', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='#404080')
    for spine in ax2.spines.values():
        spine.set_color('#404080')
    
    # Add horizontal line at max accuracy
    max_acc = max([results['accuracies'][-1] for results in results_dict.values()])
    ax2.axhline(y=max_acc, color='#00ff88', linestyle='--', alpha=0.5, linewidth=1)
    
    # ==================== PLOT 3: Training Time Bar Chart ====================
    ax3 = fig.add_subplot(gs[0, 2], facecolor='#1a1a2e')
    
    names = list(results_dict.keys())
    times = [results_dict[name]['time'] for name in names]
    bar_colors = [approach_colors.get(name, '#ffffff') for name in names]
    
    bars = ax3.barh(names, times, color=bar_colors, edgecolor='white', linewidth=1.5, height=0.6)
    
    # Add time labels on bars
    for bar, t in zip(bars, times):
        width = bar.get_width()
        ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2., 
                f'{t:.1f}s',
                ha='left', va='center', color='white', fontsize=11, fontweight='bold')
    
    ax3.set_xlabel('Training Time (seconds)', fontsize=12, color='white')
    ax3.set_title('Training Time', fontsize=14, fontweight='bold', color='white', pad=10)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='#404080', axis='x')
    ax3.set_xlim(0, max(times) * 1.3)
    for spine in ax3.spines.values():
        spine.set_color('#404080')
    
    # ==================== PLOT 4: Final Accuracy Comparison ====================
    ax4 = fig.add_subplot(gs[1, 2], facecolor='#1a1a2e')
    
    final_accs = [results_dict[name]['accuracies'][-1] for name in names]
    
    bars = ax4.barh(names, final_accs, color=bar_colors, edgecolor='white', linewidth=1.5, height=0.6)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, final_accs):
        width = bar.get_width()
        ax4.text(width - 2, bar.get_y() + bar.get_height()/2., 
                f'{acc:.1f}%',
                ha='right', va='center', color='white', fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Final Test Accuracy (%)', fontsize=12, color='white')
    ax4.set_title('Final Accuracy', fontsize=14, fontweight='bold', color='white', pad=10)
    ax4.tick_params(colors='white')
    ax4.grid(True, alpha=0.2, color='#404080', axis='x')
    ax4.set_xlim(min(final_accs) - 5, 100)
    for spine in ax4.spines.values():
        spine.set_color('#404080')
    
    # ==================== PLOT 5: Efficiency Analysis ====================
    ax5 = fig.add_subplot(gs[2, 0], facecolor='#1a1a2e')
    
    # Calculate accuracy per second
    efficiency = [results_dict[name]['accuracies'][-1] / results_dict[name]['time'] 
                  for name in names]
    
    bars = ax5.bar(range(len(names)), efficiency, color=bar_colors, 
                  edgecolor='white', linewidth=1.5, width=0.6)
    ax5.set_xticks(range(len(names)))
    ax5.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9, color='white')
    ax5.set_ylabel('Accuracy % / Second', fontsize=12, color='white')
    ax5.set_title('Training Efficiency', fontsize=14, fontweight='bold', color='white', pad=10)
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.2, color='#404080', axis='y')
    for spine in ax5.spines.values():
        spine.set_color('#404080')
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{eff:.1f}',
                ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    
    # ==================== PLOT 6: Convergence Speed ====================
    ax6 = fig.add_subplot(gs[2, 1], facecolor='#1a1a2e')
    
    # Calculate epochs to reach 90% of final accuracy
    convergence_epochs = []
    for name in names:
        accs = results_dict[name]['accuracies']
        final_acc = accs[-1]
        threshold = 0.95 * final_acc
        epochs_to_converge = next((i+1 for i, acc in enumerate(accs) if acc >= threshold), len(accs))
        convergence_epochs.append(epochs_to_converge)
    
    bars = ax6.bar(range(len(names)), convergence_epochs, color=bar_colors,
                  edgecolor='white', linewidth=1.5, width=0.6)
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9, color='white')
    ax6.set_ylabel('Epochs to 95% Final Acc', fontsize=12, color='white')
    ax6.set_title('Convergence Speed', fontsize=14, fontweight='bold', color='white', pad=10)
    ax6.tick_params(colors='white')
    ax6.grid(True, alpha=0.2, color='#404080', axis='y')
    for spine in ax6.spines.values():
        spine.set_color('#404080')
    
    for bar, epochs in zip(bars, convergence_epochs):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{epochs}',
                ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
    
    # ==================== PLOT 7: Summary Stats Box ====================
    ax7 = fig.add_subplot(gs[2, 2], facecolor='#1a1a2e')
    ax7.axis('off')
    
    # Create summary text
    standard_acc = results_dict['Standard']['accuracies'][-1]
    standard_time = results_dict['Standard']['time']
    
    summary_text = "KEY FINDINGS\n" + "-" * 25 + "\n\n"
    
    for name, results in results_dict.items():
        if name == 'Standard':
            continue
        acc_diff = results['accuracies'][-1] - standard_acc
        time_ratio = results['time'] / standard_time
        summary_text += f"► {name}:\n"
        summary_text += f"   Accuracy: {acc_diff:+.2f}%\n"
        summary_text += f"   Speed: {time_ratio:.2f}x\n\n"
    
    summary_text += "-" * 25 + "\n"
    summary_text += "=> Standard training remains\n"
    summary_text += "    most effective for\n"
    summary_text += "    cooperative objectives."
    
    ax7.text(0.05, 0.95, summary_text, 
            transform=ax7.transAxes, 
            fontsize=11, 
            color='white',
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a4a', edgecolor='#6060a0', alpha=0.8))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f'\n💾 Visualization saved to {save_path}')
    plt.show()


def plot_training_dynamics(results_dict, save_path='training_dynamics.png'):
    """Create a detailed visualization of training dynamics.
    
    Args:
        results_dict (dict): Dictionary containing results for each approach
        save_path (str): Path to save the plot
    """
    colors = setup_plot_style()
    
    approach_colors = {
        'Standard': colors['standard'],
        'Alternating (Epoch)': colors['alternating_epoch'],
        'Alternating (Batch)': colors['alternating_batch']
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0f0f1a')
    
    fig.suptitle('Training Dynamics Analysis', fontsize=16, fontweight='bold', color='white', y=1.02)
    
    names = list(results_dict.keys())
    
    # Plot 1: Loss reduction rate (derivative of loss)
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    
    for name, results in results_dict.items():
        losses = np.array(results['losses'])
        loss_reduction = -np.diff(losses)  # Negative because we want reduction to be positive
        epochs = range(1, len(loss_reduction) + 1)
        ax1.plot(epochs, loss_reduction,
                label=name,
                color=approach_colors.get(name, '#ffffff'),
                linewidth=2,
                marker='o',
                markersize=4)
    
    ax1.axhline(y=0, color='#ff4444', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss Reduction', color='white')
    ax1.set_title('Loss Reduction per Epoch', color='white', fontweight='bold')
    ax1.legend(facecolor='#2a2a4a', edgecolor='#404080', labelcolor='white', fontsize=9)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='#404080')
    for spine in ax1.spines.values():
        spine.set_color('#404080')
    
    # Plot 2: Accuracy improvement rate
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    
    for name, results in results_dict.items():
        accs = np.array(results['accuracies'])
        acc_improvement = np.diff(accs)
        epochs = range(1, len(acc_improvement) + 1)
        ax2.plot(epochs, acc_improvement,
                label=name,
                color=approach_colors.get(name, '#ffffff'),
                linewidth=2,
                marker='s',
                markersize=4)
    
    ax2.axhline(y=0, color='#ff4444', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy Δ (%)', color='white')
    ax2.set_title('Accuracy Improvement per Epoch', color='white', fontweight='bold')
    ax2.legend(facecolor='#2a2a4a', edgecolor='#404080', labelcolor='white', fontsize=9)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='#404080')
    for spine in ax2.spines.values():
        spine.set_color('#404080')
    
    # Plot 3: Cumulative comparison
    ax3 = axes[2]
    ax3.set_facecolor('#1a1a2e')
    
    # Normalize metrics for comparison
    for name, results in results_dict.items():
        final_acc = results['accuracies'][-1]
        accs_normalized = np.array(results['accuracies']) / final_acc * 100
        epochs = range(1, len(accs_normalized) + 1)
        ax3.fill_between(epochs, 0, accs_normalized,
                        alpha=0.3,
                        color=approach_colors.get(name, '#ffffff'))
        ax3.plot(epochs, accs_normalized,
                label=f"{name} ({final_acc:.1f}%)",
                color=approach_colors.get(name, '#ffffff'),
                linewidth=2)
    
    ax3.set_xlabel('Epoch', color='white')
    ax3.set_ylabel('Progress to Final Accuracy (%)', color='white')
    ax3.set_title('Relative Progress Over Training', color='white', fontweight='bold')
    ax3.legend(facecolor='#2a2a4a', edgecolor='#404080', labelcolor='white', fontsize=9)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='#404080')
    ax3.set_ylim(0, 105)
    for spine in ax3.spines.values():
        spine.set_color('#404080')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f'💾 Training dynamics saved to {save_path}')
    plt.show()


def print_summary(results_dict):
    """Print summary statistics for all approaches.
    
    Args:
        results_dict (dict): Dictionary containing results for each approach
    """
    print('\n' + '═' * 70)
    print('                        📊 SUMMARY OF RESULTS')
    print('═' * 70)
    
    for name, results in results_dict.items():
        print(f'\n🔹 {name}:')
        print(f"   Final Test Accuracy: {results['accuracies'][-1]:.2f}%")
        print(f"   Final Training Loss: {results['losses'][-1]:.4f}")
        print(f"   Total Training Time: {results['time']:.2f} seconds")
        print(f"   Time per Epoch: {results['time']/len(results['losses']):.2f} seconds")
    
    print('\n' + '═' * 70)
    print('                        📈 COMPARISON')
    print('═' * 70)
    
    # Compare to standard approach
    standard_acc = results_dict['Standard']['accuracies'][-1]
    standard_time = results_dict['Standard']['time']
    
    for name, results in results_dict.items():
        if name == 'Standard':
            continue
        
        acc_diff = results['accuracies'][-1] - standard_acc
        time_diff = results['time'] - standard_time
        time_ratio = results['time'] / standard_time
        
        emoji = '✅' if acc_diff >= 0 else '⚠️'
        print(f'\n{emoji} {name} vs Standard:')
        print(f"   Accuracy Difference: {acc_diff:+.2f}%")
        print(f"   Time Difference: {time_diff:+.2f}s ({time_ratio:.2f}x)")
    
    print('\n' + '═' * 70)
    print('\n📝 CONCLUSION:')
    print('   While alternating layer freezing can reduce computation per epoch,')
    print('   standard training typically achieves better accuracy because all')
    print('   layers cooperate toward the same objective (unlike adversarial GANs).')
    print('\n' + '═' * 70)


def run_experiments(num_epochs=15, learning_rate=0.001, batch_size=128, 
                   device='cpu'):
    """Run all experiments and compare results.
    
    Args:
        num_epochs (int): Number of epochs to train each model
        learning_rate (float): Learning rate for optimization
        batch_size (int): Batch size for training
        device (str): Device to run experiments on ('cpu' or 'cuda')
    """
    # Load data
    train_loader, test_loader = load_cifar10_data(batch_size=batch_size)
    
    results = {}
    
    # CIFAR-10: 32x32x3 = 3072 input features, 10 classes
    input_size = 3072
    
    # Experiment 1: Standard Training
    print('\n' + '═' * 70)
    print('🔬 EXPERIMENT 1: STANDARD TRAINING')
    print('═' * 70)
    print('   Training all layers at every step...')
    model_standard = SimpleNN(input_size=input_size)
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
    print('\n' + '═' * 70)
    print('🔬 EXPERIMENT 2: ALTERNATING TRAINING (EPOCH-LEVEL)')
    print('═' * 70)
    print('   Alternating between first and second half at each epoch...')
    model_alternating = SimpleNN(input_size=input_size)
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
    print('\n' + '═' * 70)
    print('🔬 EXPERIMENT 3: ALTERNATING TRAINING (BATCH-LEVEL)')
    print('═' * 70)
    print('   Alternating between first and second half at each batch...')
    model_alternating_batch = SimpleNN(input_size=input_size)
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
    plot_training_dynamics(results)
    
    return results


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n🖥️  Using device: {device}')
    print('═' * 70)
    print('   ALTERNATING LAYER FREEZING EXPERIMENT')
    print('   Comparing GAN-inspired training on CIFAR-10')
    print('═' * 70)
    
    # Run experiments
    results = run_experiments(
        num_epochs=15,
        learning_rate=0.001,
        batch_size=128,
        device=device
    )
    
    print('\n✅ Experiments completed successfully!')
