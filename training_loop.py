"""Training loops for standard and alternating optimization approaches.

This module implements different training strategies:
1. Standard training: Update all parameters at each step
2. Alternating training: Freeze one half of the network while updating the other,
   alternating between halves
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy on test set.
    
    Args:
        model (nn.Module): The neural network model to evaluate
        test_loader (DataLoader): DataLoader for test data
        device (str): Device to run evaluation on ('cpu' or 'cuda')
        
    Returns:
        float: Accuracy as a percentage (0-100)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def train_standard(model, train_loader, test_loader, num_epochs=10, 
                   learning_rate=0.001, device='cpu', verbose=True):
    """Train model using standard approach (update all parameters at each step).
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to run training on ('cpu' or 'cuda')
        verbose (bool): Whether to print progress
        
    Returns:
        tuple: (losses, accuracies, training_time)
            - losses (list): Training loss for each epoch
            - accuracies (list): Test accuracy for each epoch
            - training_time (float): Total training time in seconds
    """
    model.to(device)
    model.unfreeze_all()  # Ensure all parameters are trainable
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    accuracies = []
    
    start_time = time.time()
    
    epoch_iterator = tqdm(range(num_epochs), desc='   Standard', 
                         disable=not verbose, leave=True)
    
    for epoch in epoch_iterator:
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Evaluate on test set
        accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)
        
        # Update progress bar
        epoch_iterator.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy:.1f}%'
        })
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f'   ✓ Completed in {training_time:.2f}s | Final Acc: {accuracies[-1]:.2f}%')
    
    return losses, accuracies, training_time


def train_alternating(model, train_loader, test_loader, num_epochs=10,
                     learning_rate=0.001, device='cpu', switch_frequency=1,
                     verbose=True):
    """Train model using alternating approach (freeze one half while updating the other).
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to run training on ('cpu' or 'cuda')
        switch_frequency (int): How often to switch between halves (in epochs)
        verbose (bool): Whether to print progress
        
    Returns:
        tuple: (losses, accuracies, training_time)
            - losses (list): Training loss for each epoch
            - accuracies (list): Test accuracy for each epoch
            - training_time (float): Total training time in seconds
    """
    model.to(device)
    model.unfreeze_all()  # Start with all parameters trainable
    
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    start_time = time.time()
    
    epoch_iterator = tqdm(range(num_epochs), desc='   Alt-Epoch', 
                         disable=not verbose, leave=True)
    
    for epoch in epoch_iterator:
        model.train()
        
        # Determine which half to train based on epoch
        # Alternate between first half and second half
        train_first_half = (epoch // switch_frequency) % 2 == 0
        
        if train_first_half:
            # Train first half, freeze second half
            model.unfreeze_first_half()
            model.freeze_second_half()
            half_name = "1st"
        else:
            # Train second half, freeze first half
            model.freeze_first_half()
            model.unfreeze_second_half()
            half_name = "2nd"
        
        # Recreate optimizer with only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Evaluate on test set
        accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)
        
        # Update progress bar
        epoch_iterator.set_postfix({
            'half': half_name,
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy:.1f}%'
        })
    
    training_time = time.time() - start_time
    
    # Unfreeze all parameters at the end
    model.unfreeze_all()
    
    if verbose:
        print(f'   ✓ Completed in {training_time:.2f}s | Final Acc: {accuracies[-1]:.2f}%')
    
    return losses, accuracies, training_time


def train_alternating_batch(model, train_loader, test_loader, num_epochs=10,
                           learning_rate=0.001, device='cpu', verbose=True):
    """Train model using batch-level alternating (switch every batch instead of every epoch).
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to run training on ('cpu' or 'cuda')
        verbose (bool): Whether to print progress
        
    Returns:
        tuple: (losses, accuracies, training_time)
            - losses (list): Training loss for each epoch
            - accuracies (list): Test accuracy for each epoch
            - training_time (float): Total training time in seconds
    """
    model.to(device)
    model.unfreeze_all()
    
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    start_time = time.time()
    
    epoch_iterator = tqdm(range(num_epochs), desc='   Alt-Batch', 
                         disable=not verbose, leave=True)
    
    for epoch in epoch_iterator:
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Alternate between halves at each batch
            train_first_half = batch_idx % 2 == 0
            
            if train_first_half:
                model.unfreeze_first_half()
                model.freeze_second_half()
            else:
                model.freeze_first_half()
                model.unfreeze_second_half()
            
            # Create optimizer for current trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=learning_rate)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Evaluate on test set
        accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)
        
        # Update progress bar
        epoch_iterator.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy:.1f}%'
        })
    
    training_time = time.time() - start_time
    
    # Unfreeze all parameters at the end
    model.unfreeze_all()
    
    if verbose:
        print(f'   ✓ Completed in {training_time:.2f}s | Final Acc: {accuracies[-1]:.2f}%')
    
    return losses, accuracies, training_time
