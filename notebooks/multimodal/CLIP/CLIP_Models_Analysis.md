# File Overview
This notebook/script performs a comprehensive analysis of CLIP-based multimodal models for fake news detection on a Reddit dataset. The main tasks include data preparation, feature extraction, model evaluation, attention analysis, and visualization.

## 1. Environment Setup

- Imports standard Python libraries such as `pandas`, `numpy`, `torch`, `torchvision`, `matplotlib`, and `PIL`.
- Ensures TensorFlow is disabled.
- Sets the computation device to GPU (CUDA), CPU, or Apple MPS depending on the environment.
- Loads pretrained models from Hugging Face Transformers:
  - `CLIPModel` and `CLIPTokenizer`
  - `BertModel` and `BertTokenizer`

## 2. Data Loading and Preprocessing

- Defines file paths for training, validation, and test CSV datasets.
- Loads datasets using `pandas.read_csv`.
- Filters out corrupted rows using predefined text files containing corrupted indices.
- Adds a zero-padded `image_num` column for mapping images to dataset rows.

## 3. Dataset and DataLoader

- Implements a custom `RedditDataset` class to return `(text, image, label)` tuples.
- Defines data transformations for images:
  - Training: Resize, random horizontal flip, rotation, normalization.
  - Validation/Test: Resize and normalization.
- Defines a custom `collate_fn` to batch texts, images, and labels.
- Creates `DataLoader` objects for training, validation, and test datasets.

## 4. CLIP Feature Extraction and Cosine Similarity Analysis

- Loads pretrained CLIP model (`clip-vit-base-patch32`) and tokenizer.
- For each training batch:
  - Computes text features (`fCLIP_T`) and image features (`fCLIP_I`).
  - Calculates cosine similarity between text and image embeddings.
- Saves the cosine similarity values to a `.npy` file.
- Plots the distribution of cosine similarities before and after standardization using a sigmoid function.

## 5. Modality-Wise CLIP Model

- **Projection Head**: Two-layer MLP with batch normalization, ReLU activations, and dropout.
- **Modality-Wise Attention**: Learns attention weights across three modalities (text, image, multimodal). Aggregates features weighted by attention.
- **Classification Head**: Fully connected layers producing final binary logits.
- **FND_CLIP Model**:
  - Uses ResNet101 for image features.
  - Uses BERT for text features.
  - Uses CLIP for multimodal features.
  - Combines all features through projection heads and modality-wise attention.
  - Computes cosine similarity between CLIP text and image embeddings to weight multimodal features.
  - Produces final logits for fake/real news classification.

## 6. Model Loading and Evaluation

- Loads pretrained weights for the FND_CLIP model.
- Sets the model to evaluation mode.
- Extracts and plots training and validation loss curves (smoothed training loss vs validation loss).


## 7. Attention Weight Analysis (Modality-Wise)

- Computes attention weights for text, image, and multimodal features on the test dataset.
- Saves attention weights and predicted/ground truth labels to `.npy` files.
- Plots histograms of attention weight distributions for real vs fake news for each modality.
- Computes and prints basic statistics (mean attention weights) per modality.

## 8. QKV-Attention CLIP Model

- **QKV Attention**:
  - Replaces modality-wise attention with a query-key-value (QKV) multi-head attention module.
  - Aggregates text, image, and multimodal features as tokens in a single sequence.
  - Outputs attended features averaged across tokens.
- The remaining components (Projection Heads, Classification Head, FND_CLIP structure) are similar to the modality-wise attention model.
- Loads pretrained weights for the QKV-attention model.
- Extracts and plots training/validation loss curves.

## 9. QKV Attention Weight Analysis

- Computes attention matrices from the QKV attention module for all test samples.
- Separates attention matrices based on predicted labels (real/fake).
- Computes average attention weights across batches and heads.
- Visualizes the average attention matrices as heatmaps for real and fake news using `seaborn`.
