# MobileNetV2 Compression on CIFAR-10
This repository contains the full code and methodology for establishing a MobileNetV2 baseline on the CIFAR-10 dataset and applying a custom, iterative Mixed-Compression Pipeline (Magnitude Pruning + Per-Layer Linear Quantization) to optimize the model for edge deployment.

The project demonstrates a successful trade-off, achieving a 7.59x Total Compression Ratio with a final accuracy of 90.25%, or a 5.32x Total Compression Ratio with a final accuracy of 92.25%.

# Key Results and Reproducibility
| Run Label                  | Total Sparsity (%) | Quant. Bits | Final Acc. (%) | Conservative CR (×) | Realistic Total CR (×) |
|----------------------------|------------------|------------|----------------|-------------------|------------------------|
| Baseline                   | 0.0              | 32         | 92.86          | 1.00              | 1.00                   |
| 1. Pruned 40% (FP32)       | 40.0             | 32         | 92.75          | 1.26              | 1.55                   |
| 2. Pruned 40% (INT8)       | 40.0             | 8          | 92.63          | 2.83              | 4.92                   |
| 3. Pruned 47% (FP32)       | 47.0             | 32         | 92.26          | 1.38              | 1.73                   |
| 4. Pruned 47% (INT8)       | 47.0             | 8          | 92.25          | 2.96              | 5.32                   |
| 5. Pruned 55% (FP32)       | 55.0             | 32         | 91.52          | 1.54              | 2.00                   |
| 6. Pruned 55% (INT8)       | 55.0             | 8          | 91.56          | 3.14              | 5.94                   |
| 7. Pruned 60% (FP32)       | 60.0             | 32         | 91.56          | 1.66              | 2.21                   |
| 8. Pruned 60% (INT8)       | 60.0             | 8          | 91.63          | 3.27              | 6.41                   |
| 9. Pruned 70% (FP32)       | 70.0             | 32         | 90.25          | 1.97              | 2.80                   |
| **10. Pruned 70% (INT8)**  | **70.0**         | **8**      | **90.25**      | **3.55**          | **7.59**               |

All results are reproducible using the commands below, built upon the fixed random seed (MANUAL_SEED = 42).

# Setup
First, clone the repository:
```
git clone https://github.com/Prad-R/Mobilenet_V2_Compression.git
cd Mobilenet_V2_Compression
```

This project uses PyTorch, TorchVision, and common data science libraries. The exact environment configuration can be replicated using conda using the `environment.yml` file. The commands for the same are given below.
```
conda env create -f environment.yml
conda activate torch_env
```

Note that CUDA support is assumed to be present in the local device. If not, the code will still run, but will be significantly slower.

# Reproducing the Results

## Baseline Training
This run trains the MobileNetV2 architecture from scratch for 250 epochs to achieve the maximum possible accuracy, using the calculated CIFAR-10 mean/STD.

> [!TIP]
>  Achieve ~92.86% accuracy and save the initial checkpoint.
```
python3 train_early_stopping.py --epochs 300 --batch_size 150 --lr 0.01 --patience 20 --log_interval 500
```

> [!NOTE]
> The output file, e.g., `mobilenetv2_baseline_lr1e-02_e250_best.pth`, is required for the next phase.)

## Final Compression and Fine-Tuning
This command simulates the final successful step by loading the best baseline model, aggressively pruning it to 40% sparsity, and fine-tuning it with a low learning rate (1e-4) to recover accuracy.

```
!python3 fine_tune.py --epochs 10 --batch_size 32 --lr 1e-4 --patience 3 --log_interval 500 --pruning_sparsity 0.4 --load_path "./saved_models/from_kaggle/mobilenetv2_baseline_lr1e-02_e198_best.pth" --total_epochs_prior 198
```

Feel free to play around with the various arguments. Before selecting the sparsity, always experiment with various levels of pruning and see where the accuracy drops critically. More information can be found in `CS6886_Assn_3.ipynb`.

> [!WARNING]
> The above command aims to tune the model to a high accuracy with 40% sparsity. However, the final models obtained went through several cycles of such pruning and tuning. Refer to `CS6886_Assn_3.ipynb` for a better understanding of the methodology

# Testing and Verification

To verify the final achieved accuracy and the effectiveness of the Pruning + Quantization pipeline, use the `test.py` utility.

## Test Final Pruned/Quantized Model
This verifies the 92.25% final accuracy achieved by the 47% sparse pruning followed by quantization.

```
python3 test.py
```
To test a different model (perhaps the more aggressively pruned model), go to `test.py` and uncomment either line 10 or 13 and use the right model path.
