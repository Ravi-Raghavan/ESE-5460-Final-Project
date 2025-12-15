"""## Load Data & Setup Dataloaders"""
This section loads the multimodal dataset and prepares it for model evaluation by defining a custom PyTorch `Dataset` and wrapping it in a `DataLoader`. The dataset is responsible for reading image files from disk, applying the specified image transformations (resizing, tensor conversion, and normalization), and pairing each image with its corresponding text input and label. This ensures that every sample passed to the model contains aligned image–text data in the expected format.

The DataLoader handles batching and efficient iteration over the dataset during inference. Shuffling is disabled to preserve a consistent sample order, which is important for reproducible analysis and for correctly mapping predictions and attention outputs back to specific input samples when generating visualizations later in the script.


"""## QKV Attention Model Setup"""
This section initializes the multimodal neural network that uses a Query–Key–Value (QKV) attention mechanism to combine text and image representations. The model is constructed with separate processing pathways for each modality, followed by an attention layer that computes interactions between learned query, key, and value embeddings. This allows the model to dynamically weight information from different modalities when producing its final prediction.

The code then loads pretrained weights from a saved checkpoint, moves the model to the appropriate computation device (CPU or GPU), and switches it to evaluation mode. Importantly, the model is configured to return attention weights during the forward pass, enabling downstream inspection of how attention is distributed across text, image, and multimodal components.


"""## Sampled Attention Heatmaps"""
This section selects representative samples from the dataset and runs them through the model to extract attention weights from the QKV mechanism. The attention tensors are aggregated across heads or layers to produce a compact, interpretable representation of modality-level attention for each sample.

These aggregated attention values are then visualized as heatmaps, illustrating how strongly the model attends to text features, image features, and their combined representation when making predictions. This qualitative analysis step provides interpretability by revealing the relative contribution of each modality and helps assess whether the model is leveraging multimodal information in a meaningful way.
