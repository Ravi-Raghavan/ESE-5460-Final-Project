# FND CLIP Model with QKV Attention – Training and Evaluation

## Environment Setup
This section defines global configuration parameters such as batch size, image resolution, and attention head count, and sets up the runtime device (CPU, CUDA, or Apple MPS). It also establishes file paths for the cleaned datasets and ensures consistent execution settings across environments.

## Load Data
This section loads the cleaned training, validation, and test datasets from disk. It removes rows corresponding to known corrupted indices to prevent downstream failures during image loading or training.

Each dataset is augmented with a zero-padded `image_num` column derived from the row index. This column serves as a reliable link between tabular records and image filenames stored in the image directories.

## Create News Dataset and DataLoader
This section defines a custom PyTorch `Dataset` that returns aligned triplets of text, image, and label for each Reddit post. Images are loaded from disk, converted to RGB, and passed through modality-specific preprocessing pipelines, while text remains un-tokenized for processing inside the model.

Separate DataLoaders are created for training, validation, and testing. Training data includes augmentation to improve generalization, while validation and test data use deterministic preprocessing. A custom `collate_fn` enables batching of variable-length text inputs alongside fixed-size image tensors.

## Set Up Fake News Detection (FND) CLIP Model
This section defines the full multimodal Fake News Detection (FND) architecture. The model combines a fine-tuned ResNet-101 image encoder, a frozen BERT text encoder, and a frozen CLIP multimodal encoder to capture complementary representations.

Each modality is projected into a shared latent space using projection heads. A QKV-based multi-head attention module then performs cross-modal fusion by attending over text, image, and multimodal features. The fused representation is passed to a classification head that outputs binary fake news predictions.

## Training
This section prepares and executes the training procedure. Class proportions are computed from the training data and used to define a weighted cross-entropy loss to mitigate class imbalance.

A custom training loop iterates over epochs and mini-batches, performing forward passes, backpropagation, and parameter updates. At regular update intervals, the model is evaluated on the validation set. Both the latest model and the best-performing model (based on validation accuracy) are saved to disk for later evaluation.

## Evaluation
This section stores training and validation loss and accuracy values, visualizes learning curves, and computes final performance metrics. Training losses are optionally smoothed to highlight convergence behavior.

The best model checkpoint is reloaded and evaluated on the training, validation, and test sets. Standard classification metrics—including accuracy, precision, recall, and multiple variants of F1 score—are computed and printed. All results are saved to CSV files for reproducibility and comparison with other model variants.
