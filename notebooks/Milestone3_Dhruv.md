# Milestone 3 – Multimodal Fake News Detection with BERT + ResNet101

## Data Loading and Path Configuration
This section loads the training, validation, and test CSV files containing Reddit post metadata and labels. The code is written for execution in Google Colab, with paths pointing to Google Drive–mounted datasets. Each CSV contains textual content, labels, and indices corresponding to associated images.

## Image Index Mapping
A new `image_num` column is created by zero-padding the DataFrame indices. This establishes a consistent mapping between each textual sample and its corresponding image file on disk, enabling reliable multimodal pairing during dataset construction.

## Filtering Corrupted Samples
This section removes samples with corrupted or missing images by reading index lists from predefined text files stored on Google Drive. Rows corresponding to these indices are dropped from each split, ensuring that all remaining samples can be safely loaded during training and evaluation.

## Multimodal Dataset Definition
A custom PyTorch `Dataset` class is defined to return triplets of text, image, and label. Text is passed as raw strings for later tokenization, while images are loaded from disk using PIL and converted to tensors with optional transformations. This design supports joint text–image modeling.

## Image Transformations and DataLoaders
This section defines preprocessing pipelines for images. Training data includes random augmentations such as flipping and rotation to improve robustness, while validation and test data use deterministic resizing and normalization. DataLoaders are constructed to batch multimodal samples efficiently, using a custom `collate_fn` to handle variable-length text inputs.

## Device Configuration
The code automatically selects the appropriate computation device, prioritizing GPU when available in Colab and falling back to CPU otherwise. All tensors and model components are moved to this device to ensure consistent execution.

## Pretrained Multimodal Model (Frozen Encoders)
This section defines a multimodal classifier that combines frozen BERT and frozen ResNet101 encoders. BERT produces a [CLS] embedding for text, while ResNet extracts high-level visual features from images. These representations are concatenated and passed through a lightweight linear classification head, isolating the contribution of pretrained representations without fine-tuning.

## Class Imbalance Handling
Class proportions are computed from the training data and used to construct a weighted cross-entropy loss. This compensates for label imbalance and ensures that minority-class errors are penalized appropriately during optimization.

## Training Loop for Frozen Multimodal Model
This section implements the training procedure for the frozen-encoder multimodal model. Training metrics are recorded at every update, and validation is performed periodically to monitor generalization. Model checkpoints are saved during training to preserve intermediate states.

## Evaluation of Frozen Multimodal Model
After training, the model is evaluated on training, validation, and test sets. Standard classification metrics including accuracy, precision, recall, and multiple F1 scores are computed and saved to CSV files for systematic comparison.

## Fine-tuned Multimodal Model
This section defines a second multimodal architecture in which both BERT and ResNet101 are fully trainable. Unlike the frozen setup, gradients are allowed to flow through all layers, enabling the model to learn task-specific multimodal representations.

## Training Loop for Fine-tuned Multimodal Model
The fine-tuning training loop mirrors the frozen setup but updates all parameters jointly. Validation performance is monitored at regular intervals, and model checkpoints are saved to track learning progress and prevent loss of optimal states.

## Evaluation of Fine-tuned Multimodal Model
The final section evaluates the fully fine-tuned multimodal model on all dataset splits. The same metric suite is used as in earlier evaluations, allowing direct comparison between frozen and fine-tuned multimodal approaches in terms of classification performance.
