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
import argparse # Import the argument parser

# Define a single seed value (Fixed for reproducibility, not an argument)
MANUAL_SEED = 42

def set_seed(seed):
    """Sets the seed for reproducibility across CPU, CUDA, and common libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    # Ensure all CUDA ops are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# --- 2. DATA PREPARATION: Mean/STD Calculation and Split ---
# NOTE: These functions now accept parameters instead of relying on global variables.

def calculate_mean_std(dataset, batch_size):
    """Calculates the channel-wise mean and standard deviation of a dataset."""
    temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    mean = 0.
    total_images_count = 0
    
    # Calculate mean
    for images, _ in temp_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1) 
        mean += images.mean(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    
    # Calculate STD
    std = 0.
    for images, _ in temp_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        # Calculate variance and sum
        std += ((images - mean.unsqueeze(1))**2).sum([0, 2]) 
        
    std = torch.sqrt(std / (total_images_count * images.size(2))) 
    
    return mean.tolist(), std.tolist()

def load_and_prepare_data(input_size, batch_size, valid_split_ratio, manual_seed):
    """Loads, normalizes, and splits the CIFAR-10 data."""
    
    # Download RAW dataset to calculate statistics
    raw_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    
    print("Calculating CIFAR-10 Mean and STD...")
    CIFAR10_MEAN, CIFAR10_STD = calculate_mean_std(raw_train_dataset, batch_size)
    print(f"Calculated CIFAR-10 Mean: {CIFAR10_MEAN}")
    print(f"Calculated CIFAR-10 STD: {CIFAR10_STD}")

    # Define final transformations
    train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Load datasets with final transforms
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    # SPLIT THE TRAINING DATA INTO TRAINING AND VALIDATION SETS
    dataset_size = len(full_train_dataset)
    validation_size = int(valid_split_ratio * dataset_size)
    train_size = dataset_size - validation_size

    train_subset, validation_subset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(manual_seed) 
    )

    print(f"Dataset Split: Training ({train_size} images), Validation ({validation_size} images)")

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, validation_loader, test_loader

# --- 3. MODEL SETUP & UTILITIES ---

def create_mobilenetv2_from_scratch(num_classes):
    """
    Loads MobileNetV2 architecture with NO pre-trained weights and modifies the classifier.
    """
    model = models.mobilenet_v2(weights=None) 
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def load_model_weights(model, path):
    """Loads model weights from a .pth file."""
    try:
        if os.path.exists(path):
            print(f"Loading previous model weights from: {path}")
            # Get the current device the model is on
            map_location = next(model.parameters()).device
            model.load_state_dict(torch.load(path, map_location=map_location))
            return model
        else:
            print(f"Model weights not found at: {path}. Starting with random initialization.")
            return model
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("Model architecture or state dict file may be mismatched. Starting with random initialization.")
        return model

# --- 4. TRAINING & EVALUATION FUNCTIONS ---

def evaluate_model(model, data_loader, device, name="Test"):
    """
    Evaluates the model on the specified data loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'{name} Top-1 Accuracy: {accuracy:.2f}%')
    return accuracy

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

# --- 5. SAVING UTILITY (Updated to use arguments) ---

def save_training_data(model, history, final_test_accuracy, args, total_epochs_trained, path='./saved_models'):
    """Saves model weights and training history to disk."""
    
    # Use the relevant information from args for filename clarity
    epochs_trained = len(history['train_loss'])
    
    os.makedirs(path, exist_ok=True)
    
    # Create descriptive filename based on parameters
    # Note: Using load_path name to indicate pruning status if loaded from a pruned model
    prune_status = 'baseline' if not args.load_path else os.path.basename(args.load_path).replace('.pth', '').replace('mobilenetv2_pruned_', 'pruned_')
    
    filename_base = f"mobilenetv2_{prune_status}_lr{args.lr:.0e}_e{total_epochs_trained}"

    # Save Model State (Weights)
    model_path = os.path.join(path, f'{filename_base}_best.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save Training History (for plotting loss/accuracy curves)
    history_path = os.path.join(path, f'{filename_base}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
        
    # Save Final Baseline Accuracy
    accuracy_path = os.path.join(path, f'{filename_base}.txt')
    with open(accuracy_path, 'w') as f:
        f.write(f"Final Test Top-1 Accuracy: {final_test_accuracy:.2f}%\n")
        
    print(f"\nModel weights saved to: {model_path}")
    print(f"Training history saved to: {history_path}")

# --- 6. EXECUTION BLOCK (Refactored to use argparse) ---

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='MobileNetV2 Training and Fine-Tuning Script')
    
    # Core Training Parameters
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--log_interval', type=int, default=500, help='How many batches to wait before logging training status.')
    
    # Pruning/Loading/Continuation Parameters
    parser.add_argument('--load_path', type=str, default='', help='Path to a saved model state dictionary to load and continue training/pruning.')
    parser.add_argument('--total_epochs_prior', type=int, default=0, help='Total number of epochs already run on the loaded model.')
    
    args = parser.parse_args()
    
    # --- Dynamic Configuration ---
    set_seed(MANUAL_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fixed parameters (from the original script)
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
    
    # Load model weights if a path is provided
    if args.load_path:
        model = load_model_weights(model, args.load_path)

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
    
    # 6.4 Final Evaluation (The best weights are already loaded back into the model)
    final_test_accuracy = evaluate_model(model, test_loader, device, name="FINAL TEST SET")
    
    # 6.5 Saving
    # Calculate total epochs trained (Prior + New)
    total_epochs_trained = args.total_epochs_prior + len(training_history['train_loss'])

    save_training_data(model, training_history, final_test_accuracy, args, total_epochs_trained)