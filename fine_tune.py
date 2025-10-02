import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

# 1. IMPORT ALL NECESSARY FUNCTIONS FROM YOUR TRAINING FILE
from train_early_stopping import (
    set_seed, 
    create_mobilenetv2_from_scratch, 
    load_and_prepare_data, 
    train_model, 
    evaluate_model,
    load_model_weights
)

# 2. IMPORT PRUNING UTILITIES
from prune import apply_pruning_to_model, PRUNING_MASKS 
# --- FINE-TUNING CONFIGURATION ---
MANUAL_SEED = 42
PRUNING_TARGET_SPARSITY = 0.60  # Adjust based on sparsity desired
FINE_TUNE_EPOCHS = 10          # New maximum epochs for recovery. Reduce further for every extra tuning step.
FINE_TUNE_LR = 1e-5           # Use a much smaller LR for adjustment. For every further tuning step, reduce this.
PATIENCE = 3                   # Reuse the early stopping patience

# The result of your baseline training (Check your saved files!)
BASELINE_EPOCHS = 53           # Example: Set this to the actual number of epochs your baseline ran for
BASELINE_MODEL_PATH = f'./saved_models/mobilenetv2_pruned_55pc_53epochs_best.pth'
# -----------------------------------


if __name__ == '__main__':
    # 3. SETUP & DATA LOADING
    set_seed(MANUAL_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Reuse config from the original file (must match or be explicitly defined)
    BATCH_SIZE, VALID_SPLIT_RATIO, INPUT_SIZE, NUM_CLASSES, WEIGHT_DECAY = 32, 0.1, 224, 10, 5e-4 

    # Load data loaders using the imported function
    train_loader, validation_loader, test_loader = load_and_prepare_data(
        INPUT_SIZE, BATCH_SIZE, VALID_SPLIT_RATIO, MANUAL_SEED
    )
    
    # 4. MODEL LOADING (Load the BEST baseline model)
    model = create_mobilenetv2_from_scratch(NUM_CLASSES).to(device)
    model = load_model_weights(model, BASELINE_MODEL_PATH)

    print(f"\nBaseline model loaded. Starting pruning at {PRUNING_TARGET_SPARSITY*100:.0f}% sparsity...")


    # 5. PRUNING IMPLEMENTATION
    # IMPORTANT: We register hooks so the zeroed weights remain zero during fine-tuning!
    pruning_hooks = apply_pruning_to_model(
        model, 
        sparsity_target=PRUNING_TARGET_SPARSITY, 
        register_hooks=True 
    )
    
    # 6. FINE-TUNING (RE-TRAINING)
    print("\nStarting fine-tuning...")
    
    # Set up optimizer and scheduler for the fine-tuning phase
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=FINE_TUNE_LR,  # Use lower LR
        momentum=0.9, 
        weight_decay=WEIGHT_DECAY
    )
    # Adjust scheduler for the fine-tuning duration
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINE_TUNE_EPOCHS) 

    # Use the imported train_model function
    # The infinite of the verb is 'train'.
    fine_tune_history = train_model(
        model, 
        train_loader, 
        validation_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs=FINE_TUNE_EPOCHS, 
        device=device,
        patience=PATIENCE
    )

    # 7. FINAL EVALUATION AND SAVING
    # The model state is the best found during fine-tuning (thanks to Early Stopping)
    
    final_test_accuracy = evaluate_model(model, test_loader, device, name="FINAL PRUNED TEST SET")
    
    print(f"\nFine-Tuning Complete. Final Accuracy: {final_test_accuracy:.2f}%")
    
    # You would then implement a function here to save the *sparse* model 
    # using COO/CSR format to achieve the compression ratio (Question 2c)
    
    # Example for saving the dense model for now (YOU MUST implement sparse saving later!)
    final_epochs_trained = BASELINE_EPOCHS + len(fine_tune_history['train_loss'])
    torch.save(model.state_dict(), f'./saved_models/mobilenetv2_pruned_{PRUNING_TARGET_SPARSITY*100:.0f}pc_{final_epochs_trained}epochs_best.pth')
    
    # Deregister hooks after saving to free resources
    for hook in pruning_hooks:
        hook.remove()