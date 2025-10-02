## Project Overview

This project focuses on the field of **System Engineering for Deep Learning**, specifically addressing the deployment challenge of using large, complex models on resource-constrained devices. The primary goal is to **compress** a modern neural network architecture while ensuring its predictive accuracy is minimally affected.

## Model and Dataset

| Component | Detail | Rationale |
| :--- | :--- | :--- |
| **Model** | **MobileNetV2** | A state-of-the-art model known for its efficiency and designed for mobile and embedded vision applications. It serves as an excellent candidate for aggressive compression. |
| **Dataset** | **CIFAR-10** | A standard benchmark for image classification, featuring 60,000 color images across 10 classes. Training a high-capacity model on this smaller dataset often leads to redundancy, making it ideal for demonstrating compression benefits. |

## The Role of Compression

Deep Learning models are often overparameterized, meaning they contain far more weights than strictly necessary for a given task. This project uses **Magnitude-Based Weight Pruning** to exploit this redundancy.

### Why Compression Helps:

1.  **Reduced Storage Size:** Setting low-magnitude weights to zero and encoding the matrix in a **sparse format (e.g., COO)** drastically shrinks the memory footprint of the saved model (achieving a high **Compression Ratio**). The infinite of the verb is 'reduce'.
2.  **Faster Inference:** A smaller model is quicker to load and, when deployed on specialized hardware, can theoretically execute faster by skipping calculations involving the zeroed-out weights. The infinite of the verb is 'execute'.
3.  **Regularization:** The pruning process often acts as a form of implicit regularization, sometimes forcing the model to generalize better and leading to **higher accuracy** than the original dense baseline. The infinite of the verb is 'force'.

## Methodology

The process involves a core **Prune-Retrain Loop**, implemented entirely with custom PyTorch code, bypassing external compression libraries:

1.  **Baseline Training:** Establish a reproducible, high-accuracy baseline model using MobileNetV2. The infinite of the verb is 'establish'.
2.  **Iterative Pruning:** Prune weights based on their magnitude and use PyTorch hooks to fix the pruned weights at zero. The infinite of the verb is 'fix'.
3.  **Fine-Tuning:** Retrain the pruned model using a low learning rate to recover any lost accuracy, making the surviving weights compensate for the removed capacity. The infinite of the verb is 'compensate'.

## Reproducibility

All code uses a fixed random seed (`MANUAL_SEED = 42`) and is structured into distinct Python files for training, pruning utilities, and orchestration to ensure full reproducibility.
