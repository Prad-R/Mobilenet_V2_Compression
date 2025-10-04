from quantization import linear_quantize_and_evaluate, save_quantized_model, verify_precision
from train_early_stopping import create_mobilenetv2_from_scratch, load_model_weights, load_and_prepare_data
import torch

## Loading the final models
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========
# Note that the SPARSE & QNUATIZED model can't be run directly. The models (.pth files) from the ./saved_models directory should be run as they have the weights in an acceptable format
# ===========

## For 70% sparsity, uncomment the line below
# FINAL_PRUNED_MODEL_PATH = f"./saved_models/mobilenetv2_mobilenetv2_mobilenetv2_mobilenetv2_mobilenetv2_mobilenetv2_baseline_lr1e-02_e198_best_lr1e-04_e207_best_lr5e-05_e217_best_lr5e-05_e223_best_lr5e-05_e233_best_lr5e-05_e243_best.pth"

## For 47% sparsity, uncomment the line below
FINAL_PRUNED_MODEL_PATH = f"./saved_models/mobilenetv2_mobilenetv2_mobilenetv2_baseline_lr1e-02_e198_best_lr1e-04_e207_best_lr5e-05_e217_best.pth"

final_pruned_model = create_mobilenetv2_from_scratch(NUM_CLASSES).to(device)
final_pruned_model = load_model_weights(final_pruned_model, FINAL_PRUNED_MODEL_PATH)
##

## Setting parameters for loading the data
BATCH_SIZE = 32      
INPUT_SIZE = 224      
VALID_SPLIT_RATIO = 0.1 # 10% of the training data will be used for validation
MANUAL_SEED = 42

train_loader, validation_loader, test_loader = load_and_prepare_data(
    INPUT_SIZE, BATCH_SIZE, VALID_SPLIT_RATIO, MANUAL_SEED
)
##

## Measuring the performance of the final model
final_quant_accuracy, quant_metadata = linear_quantize_and_evaluate(
    model=final_pruned_model, 
    data_loader=test_loader, 
    device=device, 
    name="Final Pruned + INT8 Model"
)
##

print(f"\nAccuracy after Layer-Wise INT8 Quantization: {final_quant_accuracy:.2f}%")

## Define the final save path
FINAL_SAVE_PATH = './saved_models/final/mobilenetv2_pruned_47_and_quantized_int8.pth'

## Call the saving function
save_quantized_model(
    model=final_pruned_model, # The model still holds the original FP32 sparse weights
    quant_metadata=quant_metadata, # The calculated Scales/ZPs (metadata)
    save_path=FINAL_SAVE_PATH
)

print("\n===========")
print("Note that the SPARSE & QNUATIZED model can't be run directly. The models (.pth files) from the ./saved_models directory should be run as they have the weights in an acceptable format.")
print("===========")