# Milestone 4: FND CLIP Model with Modality Attention - Analysis

## Environment Setup
This section initializes the runtime configuration for the analysis pipeline. It sets file paths, establishes reproducibility settings, and selects the appropriate computation device depending on the execution environment. The setup ensures that subsequent analysis runs consistently across systems.

## Load Data
This section loads the cleaned training, validation, and test datasets used for analysis. It removes known corrupted samples using predefined index files to guarantee that all remaining rows are valid and aligned with available image data.

Each dataset is augmented with a zero-padded `image_num` identifier derived from the row index. This identifier provides a direct and reliable mapping between tabular entries and corresponding image files stored on disk, ensuring consistency across text, image, and label modalities.

## Create News Dataset and DataLoader
This section defines a custom PyTorch `Dataset` that returns aligned text, image, and label triplets for each Reddit post. Images are loaded from disk, converted to RGB format, and transformed using modality-specific preprocessing pipelines, while text is preserved as raw strings for downstream tokenization.

Separate DataLoaders are created for training, validation, and testing. Training data includes augmentation to improve generalization, while validation and test loaders use deterministic preprocessing. A custom collation function enables efficient batching of variable-length text alongside fixed-size image tensors.

## Load CLIP Text and Image Encoders
This section loads a pretrained CLIP model and its tokenizer to extract joint text and image embeddings. The CLIP encoders serve as a foundation for analyzing semantic alignment between textual captions and visual content in the dataset.

The model is moved to the appropriate device and used strictly in inference mode for feature extraction and similarity analysis.

## Analyze Cosine Similarities
This section computes cosine similarity scores between CLIP text embeddings and CLIP image embeddings across the training dataset. For each batch, text and image features are extracted independently and compared to quantify cross-modal alignment.

The resulting similarity scores are accumulated and saved for later analysis. These values provide insight into how strongly text and images agree semantically before any model development is done.

## Plot Cosine Similarities
This section visualizes the distribution of cosine similarity values before and after standardization. The similarities are normalized using their empirical mean and variance and passed through a sigmoid function to examine how weighting functions respond to raw versus standardized inputs.

The resulting plots illustrate the effect of normalization on similarity-based gating and motivate the design choices used later in multimodal feature weighting.

## Load FND CLIP Model
This section reconstructs the full Fake News Detection CLIP (FND-CLIP) architecture used during training. The model integrates a ResNet image encoder, a frozen BERT text encoder, and a frozen CLIP multimodal encoder.

Projection heads map each modality into a shared latent space, after which a modality-wise attention mechanism dynamically assigns weights to text, image, and multimodal features. A final classification head produces binary fake news predictions. Trained model weights are loaded from disk, and the model is set to evaluation mode.

## Analyze Training Loss Curves
This section loads previously saved training and validation loss values from disk. Training losses are smoothed using a rolling average to reduce noise and better reveal convergence trends.

The smoothed training loss and validation loss are plotted together, allowing inspection of optimization behavior, convergence stability, and potential overfitting.

## Modality Wise Attention Heat-Map
This section runs the trained model on the test dataset to extract modality-wise attention weights for each sample. For every prediction, the model outputs attention weights corresponding to text, image, and multimodal representations.

These weights, along with predicted and ground-truth labels, are stored for downstream interpretability analysis. The saved arrays form the basis for qualitative inspection of model behavior.

## Case 1: Text is Dominant Modality
This section analyzes correctly classified samples where text attention significantly outweighs image and multimodal attention. Such cases highlight scenarios where linguistic cues alone are sufficient for fake news detection.

Selected examples are visualized with the original image, wrapped text caption, prediction details, and a heatmap showing modality attention weights, providing interpretability into text-driven decisions.

## Case 2: Image is Dominant Modality
This section focuses on correctly classified samples where image attention dominates. These cases demonstrate situations in which visual cues are more informative than text for identifying fake news.

Each example is visualized with its image, caption, predicted and true labels, and a modality attention heatmap to illustrate how the model prioritizes visual information.

## Case 3: Multimodal Features are Needed
This section examines samples where neither text nor image alone dominates, and the multimodal representation is strongly present. These cases indicate that joint reasoning over text and image is necessary for correct classification.

Visualizations combine the image, caption, prediction metadata, and modality attention heatmaps, showcasing how the model leverages cross-modal interactions to resolve ambiguous or complex examples.
