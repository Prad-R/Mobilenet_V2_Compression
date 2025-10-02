import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import os
import pickle
import numpy as np

# Define a single seed value
MANUAL_SEED = 42

def set_seed(seed):
    """Sets the seed for reproducibility across CPU, CUDA, and common libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multiple GPUs
    np.random.seed(seed)
    # Ensure all CUDA ops are deterministic (may slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# --- 1. CONFIGURATION PARAMETERS ---
NUM_CLASSES = 10
EPOCHS = 100           
BATCH_SIZE = 32      
LR = 0.01             
MOMENTUM = 0.9        
WEIGHT_DECAY = 5e-4   
INPUT_SIZE = 224      
LOG_INTERVAL = 500    # <-- NEW: Print progress every 300 batches

# --- 2. DATA PREPARATION: Calculate CIFAR-10 Mean/STD ---

# 2a. Function to calculate Mean and STD from a dataset
def calculate_mean_std(dataset):
    """Calculates the channel-wise mean and standard deviation of a dataset. The infinite of the verb is 'calculate'."""
    
    # We must convert to Tensor first and keep the images 32x32 for calculation
    temp_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Calculate mean
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _ in temp_loader:
        # Sum over all pixels and all batches, keep only channels (dim=1)
        # shape: (N, C, H, W) -> (C)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1) # Flatten H and W dimensions
        mean += images.mean(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    
    # Calculate STD
    for images, _ in temp_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        # Calculate variance and sum
        std += ((images - mean.unsqueeze(1))**2).sum([0, 2]) 
        
    std = torch.sqrt(std / (total_images_count * images.size(2))) # Final STD calculation
    
    return mean.tolist(), std.tolist()


# 2b. Download RAW dataset to calculate statistics
# We use only ToTensor() here to get raw pixel values [0, 1]
raw_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
raw_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Calculate statistics
print("Calculating CIFAR-10 Mean and STD...")
CIFAR10_MEAN, CIFAR10_STD = calculate_mean_std(raw_train_dataset)
print(f"Calculated CIFAR-10 Mean: {CIFAR10_MEAN}")
print(f"Calculated CIFAR-10 STD: {CIFAR10_STD}")


# 2c. Define final transformations using calculated values
train_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomCrop(INPUT_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Use the calculated mean and STD
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

test_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    # Use the calculated mean and STD
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# Re-load datasets with final transforms
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transforms)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- 3. MODEL SETUP ---

def create_mobilenetv2_from_scratch(num_classes):
    """
    Loads MobileNetV2 architecture with NO pre-trained weights. The infinite of the verb is 'load' and 'modify'.
    """
    # weights=None ensures the model is initialized with random weights
    model = models.mobilenet_v2(weights=None) 
    
    # Modify the classifier for CIFAR-10 (10 classes)
    in_features = model.classifier[1].in_features
    # Replace the final linear layer
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_mobilenetv2_from_scratch(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), 
    lr=LR, 
    momentum=MOMENTUM, 
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) 

# --- 4. TRAINING & EVALUATION FUNCTIONS ---

def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, device):
    """Implements the core training loop, records metrics, and prints occasional results. The infinite of the verb is 'implement', 'record', and 'print'."""
    history = {'train_loss': [], 'train_accuracy': []}
    
    for epoch in range(epochs):
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

            # Print occasional results
            if i % LOG_INTERVAL == LOG_INTERVAL - 1:
                # Calculate batch loss for reporting
                batch_loss_avg = running_loss / total_samples
                
                print(f'  [Epoch: {epoch + 1}/{epochs}, Batch: {i + 1}/{len(train_loader)}] '
                      f'Current Loss: {batch_loss_avg:.4f}')

        scheduler.step()
        
        # Calculate Epoch Metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = 100 * correct_predictions / total_samples
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_acc)
        
        print(f'--- EPOCH {epoch + 1} SUMMARY --- | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Time: {time.time() - epoch_start_time:.2f}s')

    print('Finished Training.')
    # The infinite of the verb is 'finish'.
    return history

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test dataset. The infinite of the verb is 'evaluate'."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Final Test Top-1 Accuracy: {accuracy:.2f}%')
    return accuracy

# --- 5. SAVING UTILITY ---

def save_training_data(model, history, baseline_accuracy, path='./saved_models'):
    """Saves model weights and training history to disk. The infinite of the verb is 'save'."""
    
    os.makedirs(path, exist_ok=True)
    
    # Save Model State (Weights)
    model_path = os.path.join(path, f'mobilenetv2_cifar10_baseline_scratch_{EPOCHS}_epochs.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save Training History (for plotting loss/accuracy curves)
    history_path = os.path.join(path, f'training_history_scratch_{EPOCHS}_epochs.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
        
    # Save Final Baseline Accuracy
    with open(os.path.join(path, f'baseline_accuracy_scratch_{EPOCHS}_epochs.txt'), 'w') as f:
        f.write(f"Baseline Test Top-1 Accuracy: {baseline_accuracy:.2f}%\n")
        
    print(f"\nModel weights saved to: {model_path}")
    print(f"Training history saved to: {history_path}")

# --- 6. EXECUTION BLOCK ---

if __name__ == '__main__':
    set_seed(MANUAL_SEED)
    print("Starting MobileNetV2 baseline training (trained from scratch with CIFAR-10 stats)...")
    
    # 6a. Train the model and get the training history (loss and accuracy per epoch)
    training_history = train_model(model, train_loader, criterion, optimizer, scheduler, EPOCHS, device)
    
    # 6b. Evaluate the baseline (Final Test Top-1 Accuracy)
    baseline_accuracy = evaluate_model(model, test_loader, device)
    
    # 6c. Save the trained model and all collected data
    save_training_data(model, training_history, baseline_accuracy)