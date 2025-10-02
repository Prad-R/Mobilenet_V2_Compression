import torch
import torch.nn as nn

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

# In your prune.py file:

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