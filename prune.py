import torch
import torch.nn as nn
import os

# --- Global Storage for Masks (Crucial for the custom implementation) ---
# We use a dictionary to store the mask for each module.
PRUNING_MASKS = {}

def prune_by_magnitude(module: nn.Module, sparsity_target: float):
    """
    Calculates the magnitude threshold and creates a binary mask for the weight tensor.
    The mask is stored globally and applied to the weights immediately.
    """
    # Only prune layers with a 'weight' attribute (Conv2d, Linear)
    if not hasattr(module, 'weight') or module.weight is None:
        return

    # 1. Prepare Weight Data
    weight = module.weight.data
    
    # 2. Calculate the magnitude threshold (tau)
    with torch.no_grad():
        # Calculate the number of elements to keep (non-zero)
        total_elements = weight.numel()
        elements_to_keep = int(total_elements * (1.0 - sparsity_target))
        
        # Calculate the number of elements to prune (zero)
        elements_to_prune = total_elements - elements_to_keep
        
        if elements_to_prune <= 0:
            print(f"Skipping pruning: Target sparsity {sparsity_target*100:.1f}% is too low.")
            return

        # Find the threshold (tau) for the k-th smallest magnitude
        flat_abs_weights = torch.abs(weight).flatten()
        
        # Use torch.topk to find the magnitude threshold
        # We find the magnitude of the (elements_to_prune)-th smallest element
        threshold, _ = torch.topk(flat_abs_weights, elements_to_prune, largest=False)
        threshold = threshold[-1] # The threshold is the magnitude of the last element in the top-k list
        
        # 3. Create the binary mask (M)
        # M[i] = 0 if |w[i]| <= threshold, 1 otherwise.
        mask = torch.where(
            torch.abs(weight) > threshold, 
            torch.tensor(1.0, device=weight.device), 
            torch.tensor(0.0, device=weight.device)
        )
        
        # 4. Apply the mask permanently to the weights
        module.weight.data.mul_(mask)
        
        # 5. Store the mask for gradient management (fine-tuning)
        # We use the module name as the key.
        PRUNING_MASKS[module] = mask
        
        actual_sparsity = 1.0 - (mask.sum().item() / total_elements)
        print(f"Pruned {module.__class__.__name__} | Target: {sparsity_target*100:.1f}%, Actual: {actual_sparsity*100:.2f}%, Threshold: {threshold:.6f}")

### 2. Gradient Masking Hook (Prune-Retrain Loop)
def apply_pruning_mask_to_grad(module, grad_input, grad_output):
    """
    Hook function: Multiplies the gradient by the stored mask.
    """
    # This check ensures the gradient exists and the module has a mask
    if module in PRUNING_MASKS and module.weight.grad is not None:
        mask = PRUNING_MASKS[module].to(module.weight.grad.device)
        # Zero out the gradient for all weights that were pruned
        module.weight.grad.data.mul_(mask)

# To apply pruning
def apply_pruning_to_model(model: nn.Module, sparsity_target: float, register_hooks: bool = False):
    """
    Applies magnitude pruning to all convolutional and linear layers.
    Hooks are registered ONLY if register_hooks is True.
    """
    global PRUNING_MASKS
    PRUNING_MASKS = {} # Clear masks for a fresh pruning run
    
    hooks = []

    print(f"\n--- Applying Pruning (Sparsity Target: {sparsity_target*100:.1f}%) ---")
    
    # Use model.named_modules() to iterate over all layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Skip the first convolutional layer (often too critical) or BN layers
            if "bn" in name.lower() or name == 'features.0.0': 
                continue 
            
            # Prune the weights and store the mask
            prune_by_magnitude(module, sparsity_target)
            
            # --- MODIFICATION START: Only register hooks if requested ---
            if register_hooks:
                # Register the hook to manage gradients during fine-tuning
                hook = module.register_backward_hook(apply_pruning_mask_to_grad)
                hooks.append(hook)
            # --- MODIFICATION END ---
            
    if register_hooks:
        print(f"--- Pruning Complete. {len(hooks)} gradient hooks registered for fine-tuning. ---")
    else:
        print("--- Pruning Complete. No gradient hooks registered (for prune-only evaluation). ---")
        
    return hooks # Returns an empty list if hooks were not registered

# To verify sparsity
def verify_comparative_sparsity(baseline_path: str, pruned_path: str) -> dict:
    """
    Loads baseline and pruned models to calculate the true sparsity by comparing 
    the set of non-zero weights in the baseline against the set of zero weights 
    in the final model.
    """
    if not os.path.exists(baseline_path):
        return {"error": f"Baseline model not found at {baseline_path}"}
    if not os.path.exists(pruned_path):
        return {"error": f"Pruned model not found at {pruned_path}"}

    # Load state dictionaries
    baseline_state = torch.load(baseline_path, map_location='cpu')
    pruned_state = torch.load(pruned_path, map_location='cpu')

    total_weights_to_prune = 0  # Total weights that were NON-ZERO in baseline
    actual_pruned_weights = 0   # Total weights that ARE ZERO in the pruned model
    
    # We iterate over the pruned model's state dictionary keys
    for name, pruned_param in pruned_state.items():
        # Ensure the parameter exists in the baseline and is a weight/bias
        if name in baseline_state and ('weight' in name or 'bias' in name):
            
            baseline_param = baseline_state[name]
            
            # --- Step 1: Identify weights that were non-zero in the baseline ---
            # Create a mask for weights that were originally part of the active network
            # (We only count non-zero elements in the baseline as eligible for pruning)
            baseline_non_zero_mask = (torch.abs(baseline_param) > 1e-9) 
            
            # Sum of elements eligible for pruning
            total_eligible_weights = baseline_non_zero_mask.sum().item()
            total_weights_to_prune += total_eligible_weights
            
            # --- Step 2: Identify weights that are now zero in the pruned model ---
            # Create a mask for weights that are near-zero (our pruned weights)
            pruned_zero_mask = (torch.abs(pruned_param) < 1e-9)
            
            # --- Step 3: Find the INTERSECTION (Weights that were NON-ZERO, but are NOW ZERO) ---
            # We must only count weights that were eligible AND are now zero.
            truly_pruned_in_this_layer = (baseline_non_zero_mask & pruned_zero_mask).sum().item()
            actual_pruned_weights += truly_pruned_in_this_layer
    
    if total_weights_to_prune == 0:
        return {"error": "No eligible weights found in baseline model (unexpected)"}

    # Calculate final true sparsity
    true_sparsity = actual_pruned_weights / total_weights_to_prune
    
    return {
        "true_sparsity": true_sparsity,
        "total_eligible_weights": total_weights_to_prune,
        "actual_pruned_weights": actual_pruned_weights,
        "compression_ratio": 1.0 / (1.0 - true_sparsity)
    }