## Training and Validation Loss Analysis
This section analyzes the training dynamics of the fine-tuned BERT model by extracting loss values recorded during training and evaluation. The script loads the `trainer_state.json` file produced by the Hugging Face Trainer, which contains a detailed log of training and validation metrics at different optimization steps.

From the training log history, the code separates training loss and validation loss entries and records their corresponding step counts. These values are then plotted to visualize how the model’s loss evolves over time on both the training and validation sets. Comparing these curves helps assess convergence behavior, training stability, and potential overfitting or underfitting during fine-tuning.
