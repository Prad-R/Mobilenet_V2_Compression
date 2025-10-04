import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os
import pickle
import numpy as np
import copy 
import argparse

from train_early_stopping import load_model_weights, evaluate_model, set_seed, calculate_mean_std, load_and_prepare_data, create_mobilenetv2_from_scratch, save_training_data

# --- 1. IMPORT PRUNING UTILITIES ---
# NOTE: This line assumes you have a file named 'prune.py' in the same directory
# that contains the necessary functions.
from prune import apply_pruning_to_model 

# Define a single seed value (Fixed for reproducibility)
MANUAL_SEED = 42

# Add this function near the top of fine_tune.py

def check_model_sparsity(model):
    """Calculates and returns the absolute sparsity of a model's weights."""
    total_elements = 0
    total_zero_elements = 0
    for name, param in model.named_parameters():
        if 'weight' in name: # We only consider weights for this check
            total_elements += param.numel()
            total_zero_elements += (torch.abs(param.data) < 1e-8).sum().item()
    if total_elements == 0:
        return 0.0
    return total_zero_elements / total_elements

def check_state_dict_sparsity(state_dict):
    """Calculates the absolute sparsity of a model's state_dict."""
    total_elements = 0
    total_zero_elements = 0
    for name, param in state_dict.items():
        if 'weight' in name:
            total_elements += param.numel()
            total_zero_elements += (torch.abs(param.data) < 1e-8).sum().item()
    if total_elements == 0:
        return 0.0
    return total_zero_elements / total_elements

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device, patience, log_interval):
    """
    Implements the core training loop with per-epoch validation and early stopping.
    """
    history = {'train_loss': [], 'train_acc': [], 'valid_acc': []}
    
    best_valid_acc = 0.0
    epochs_no_improve = 0
    best_model_weights = None
    
    print(f"Starting training with early stopping (patience={patience})...")

    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_start_time = time.time()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for module, mask in PRUNING_MASKS.items():
                    module.weight.data.mul_(mask)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % log_interval == 0:
                batch_loss_avg = running_loss / total_samples
                print(f'  [Epoch: {epoch + 1}/{epochs}, Batch: {i + 1}/{len(train_loader)}] Current Running Loss: {batch_loss_avg:.4f}')

        scheduler.step()
        
        # Calculate Training Metrics
        train_loss = running_loss / total_samples
        train_acc = 100 * correct_predictions / total_samples
        
        # 2. Evaluation Phase (Validation)
        current_valid_acc = evaluate_model(model, valid_loader, device, name="Validation Set")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(current_valid_acc)
        
        print(f'--- EPOCH {epoch + 1} SUMMARY --- | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Valid Acc: {current_valid_acc:.2f}% | Time: {time.time() - epoch_start_time:.2f}s')

        # 3. Early Stopping Logic
        if current_valid_acc > best_valid_acc:
            best_valid_acc = current_valid_acc
            epochs_no_improve = 0
            # Save a deep copy of the best weights
            best_model_weights = copy.deepcopy(model.state_dict())

            sparsity_of_snapshot = check_state_dict_sparsity(best_model_weights)
            print(f"    *DEBUG: Sparsity of the 'best_model_weights' snapshot is: {sparsity_of_snapshot*100:.2f}%*")

            print(f"    *New best model found at Epoch {epoch + 1}. Saving weights...*")
        else:
            epochs_no_improve += 1
            print(f"    *No improvement in validation accuracy. Patience counter: {epochs_no_improve}/{patience}*")
            if epochs_no_improve >= patience:
                print(f"\nEARLY STOPPING: Validation accuracy has not improved for {patience} epochs.")
                break 

    # Load the best weights found before finishing
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    print('Finished Training.')
    return history

# --- 6. EXECUTION BLOCK (Refactored to use argparse) ---

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='MobileNetV2 Training and Fine-Tuning Script')
    
    # Core Training Parameters
    parser.add_argument('--epochs', type=int, default=10, help='Max number of epochs to train (e.g., 100 for baseline, 10 for fine-tuning).')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping.')
    parser.add_argument('--log_interval', type=int, default=500, help='How many batches to wait before logging training status.')
    
    # Pruning/Loading/Continuation Parameters
    parser.add_argument('--pruning_sparsity', type=float, default=0.0, help='Sparsity target applied in this run (0.0 to 1.0).')
    parser.add_argument('--load_path', type=str, default='', help='Path to a saved model state dictionary to load and continue training/pruning.')
    parser.add_argument('--total_epochs_prior', type=int, default=0, help='Total number of epochs already run on the loaded model.')
    
    args = parser.parse_args()
    
    # --- Dynamic Configuration ---
    set_seed(MANUAL_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fixed parameters 
    NUM_CLASSES = 10
    WEIGHT_DECAY = 5e-4
    INPUT_SIZE = 224
    VALID_SPLIT_RATIO = 0.1
    MOMENTUM = 0.9

    # 6.1 Data Loading and Preparation
    train_loader, validation_loader, test_loader = load_and_prepare_data(
        INPUT_SIZE, args.batch_size, VALID_SPLIT_RATIO, MANUAL_SEED
    )

    # 6.2 Model Setup
    model = create_mobilenetv2_from_scratch(NUM_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model) # Wrap the model
    
    # Load model weights if a path is provided
    if args.load_path:
        model = load_model_weights(model, args.load_path)
    
    # --- PRUNING EXECUTION STEP ---
    pruning_hooks = []
    if args.pruning_sparsity > 0.0:
        print(f"Applying pruning with target sparsity: {args.pruning_sparsity*100:.2f}%")
        pruning_hooks, PRUNING_MASKS = apply_pruning_to_model(
            model, 
            sparsity_target=args.pruning_sparsity, 
            # register_hooks=True 
            register_hooks=False
        )
    # -----------------------------

    # print(PRUNING_MASKS)

    # Set up Optimizer and Scheduler based on ARGUMENTS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) 

    # 6.3 Training
    print("Starting MobileNetV2 training...")
    
    training_history = train_model(
        model, 
        train_loader, 
        validation_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs=args.epochs, 
        device=device,
        patience=args.patience,
        log_interval=args.log_interval # Pass log_interval
    )

    # =================================================================
    # --- POST-TRAINING SPARSITY CHECKS ---
    print("\n--- POST-TRAINING SPARSITY CHECKS ---")

    # STEP 1: Check sparsity right after the training loop and loading the best model.
    sparsity_after_train = check_model_sparsity(model)
    print(f"[DEBUG 1] Sparsity after train_model completes: {sparsity_after_train*100:.2f}%")
    # =================================================================
    
    # 6.4 Final Evaluation
    final_test_accuracy = evaluate_model(model, test_loader, device, name="FINAL TEST SET")

    # =================================================================
    # STEP 2: Check sparsity after the final evaluation.
    sparsity_after_eval = check_model_sparsity(model)
    print(f"[DEBUG 2] Sparsity after final evaluation: {sparsity_after_eval*100:.2f}%")
    # =================================================================
    
    # Deregister hooks before saving
    for hook in pruning_hooks:
        hook.remove()
        
    # 6.5 Saving
    # Calculate total epochs trained (Prior + New)
    total_epochs_trained = args.total_epochs_prior + len(training_history['train_loss'])

    save_training_data(model, training_history, final_test_accuracy, args, total_epochs_trained)