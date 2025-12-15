# Train Fake News Detection CLIP Model

## Environment Setup
This section establishes the runtime configuration for training a multimodal fake news detection model using text and images. It defines file paths for the cleaned datasets, sets a fixed random seed for reproducibility, and determines the appropriate computation device depending on whether the code is running in Google Colab or on a local machine (CPU, CUDA, or Apple MPS).

The environment setup ensures consistent experimental conditions across runs and prepares the script for large-scale multimodal training by explicitly disabling TensorFlow and standardizing device selection.

## Load Data
This section loads the cleaned training, validation, and test datasets from CSV files. These datasets contain Reddit post titles, binary fake news labels, and metadata required to link each row to a corresponding image file.

Known corrupted rows are removed using precomputed index lists to prevent downstream failures during image loading or model training. Each dataset is then augmented with a zero-padded image identifier (`image_num`) that directly maps tabular rows to image filenames on disk, ensuring alignment between text, image, and label data.

## Create News Dataset and DataLoader
This section defines a custom PyTorch `Dataset` class that returns aligned text, image, and label triples for each sample. Images are loaded from disk, converted to RGB format, and passed through modality-specific transformations, while text is kept as raw strings for tokenization inside the model.

Separate transformation pipelines are defined for training and evaluation to introduce data augmentation during training while keeping validation and test preprocessing deterministic. Custom DataLoaders are created with a specialized `collate_fn` to batch variable-length text inputs alongside image tensors and labels, enabling efficient multimodal training.

## Set Up Fake News Detection (FND) CLIP Model
This section defines the full multimodal Fake News Detection (FND) model architecture. The model integrates three complementary encoders: a ResNet-101 image encoder, a frozen BERT text encoder, and a frozen CLIP multimodal encoder. Each modality produces high-dimensional feature representations that capture different aspects of the input data.

Projection heads reduce each modality’s feature dimensionality into a shared latent space. A modality-wise attention mechanism then learns dynamic weights over text, image, and multimodal representations, allowing the model to emphasize the most informative modalities on a per-sample basis. A final classification head maps the aggregated representation to binary fake news predictions.

## Training Loop Setup
This section prepares the optimization components required for training. Class proportions are computed from the training data and used to define a weighted cross-entropy loss function, mitigating class imbalance by penalizing errors on underrepresented classes more heavily.

An AdamW optimizer is initialized to update model parameters, balancing efficient convergence with regularization. These components together define the objective function and update rules used during training.

## Model Training Procedure
This section implements a custom training loop that iterates over multiple epochs and mini-batches. For each batch, the model performs a forward pass, computes the weighted loss, backpropagates gradients, and updates parameters. Training loss and accuracy are tracked at every step to monitor learning progress.

At regular update intervals, the model is evaluated on the validation set to compute validation loss and accuracy. Model checkpoints are saved periodically, enabling recovery and inspection of intermediate training states. This procedure provides fine-grained control over training dynamics beyond standard trainer abstractions.

## Training and Validation Diagnostics
This section visualizes training and validation behavior by plotting loss and accuracy curves collected during training. Both raw and smoothed training loss trajectories are displayed to highlight overall convergence trends while reducing noise from individual updates.

These diagnostics help identify instability, underfitting, or overfitting and provide empirical evidence of training effectiveness.

## Final Evaluation and Metrics
This section evaluates the trained model on the training, validation, and test datasets using standard classification metrics. Model logits are collected across each split and converted into predictions to compute accuracy, precision, recall, and F1 scores for both positive and negative classes.

Macro, micro, and weighted F1 scores are also reported to provide a balanced performance assessment under class imbalance. All evaluation results are saved to CSV files, enabling reproducible reporting and comparison with other models.
