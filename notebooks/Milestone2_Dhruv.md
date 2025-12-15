# Milestone 2: Image-Only Fake News Detection with ResNet101

## Data Loading and Path Configuration
This section defines file paths for the training, validation, and test CSV files and loads them into pandas DataFrames. The code is written to be portable across different machines by allowing path overrides depending on the execution environment. Each dataset contains labels and metadata corresponding to Reddit posts with associated images.

## Image Index Mapping
Here, a new `image_num` column is created by zero-padding the DataFrame index. This step establishes a deterministic mapping between each row in the CSV files and its corresponding image filename on disk, enabling reliable image retrieval during dataset construction.

## Filtering Corrupted Samples
This section removes samples known to have corrupted or missing images. The indices of such samples are read from predefined text files, and the corresponding rows are dropped from each dataset. This ensures that all remaining samples can be safely loaded and processed during training and evaluation.

## Image Dataset Definition
This section defines a custom PyTorch `Dataset` class that loads images and labels from disk based on the CSV metadata. Each item returns a single image tensor and its associated binary label. Image loading is handled with PIL, and optional transformations are applied to standardize input size and normalization.

## Image Transformations and DataLoaders
This section specifies preprocessing pipelines for training and evaluation. Training images include random augmentations such as horizontal flipping and rotation to improve generalization, while validation and test images use deterministic resizing and normalization. DataLoaders are then created to batch and shuffle the data efficiently during training and evaluation.

## Device Setup
This section selects the appropriate computation device based on the runtime environment. If running in Google Colab with GPU support, CUDA is used; otherwise, Apple MPS or CPU is selected. All subsequent model operations are performed on this device.

## Pre-trained ResNet101 Model
This section initializes a ResNet101 model pretrained on ImageNet and adapts it for binary fake news classification by replacing the final fully connected layer. All backbone parameters are frozen so that only the classification head is trained. This setup evaluates the performance of fixed visual features for the task.

## Loss Function and Optimizer
This section computes class proportions from the training data and defines a weighted cross-entropy loss to address class imbalance. An AdamW optimizer is configured to update the trainable parameters of the model during optimization.

## Training Loop for Pre-trained Model
This section implements the training loop for the frozen ResNet101 model. It performs forward and backward passes, tracks batch-level loss and accuracy, and periodically evaluates performance on the validation set. Model checkpoints are saved at regular intervals for later analysis.

## Evaluation of Pre-trained Model
This section evaluates the trained model on the training, validation, and test datasets. It computes standard classification metrics including accuracy, precision, recall, and multiple F1 variants. All results are saved to CSV files for comparison and reporting.

## Fine-tuned ResNet101 Model
This section reinitializes the ResNet101 model but enables gradient updates for all layers, allowing full end-to-end fine-tuning. The classification head remains adapted for binary output, and the same loss function and optimizer configuration are reused.

## Training Loop for Fine-tuned Model
This section mirrors the earlier training loop but now updates all ResNet parameters. The model learns task-specific visual representations through full fine-tuning, and validation performance is monitored throughout training. Updated checkpoints are saved to disk.

## Evaluation of Fine-tuned Model
This final section evaluates the fine-tuned ResNet101 model on the training, validation, and test sets. The same suite of classification metrics is computed and saved, enabling direct comparison between the frozen and fine-tuned image-only baselines.
