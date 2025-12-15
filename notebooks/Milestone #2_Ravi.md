# Milestone 2: BERT Fine-tuning

## Milestone Aim

The goal is to use a **pretrained BERT-Base, Uncased** model and **fine-tune it on the r/Fakeeddit dataset**.

This work presents evaluation results on:
- Pretrained BERT
- Fine-Tuned BERT

## Script Sanity Check

Please ensure your directory is structured as follows

```text
cleaned_data/
├── test_5k.csv
├── train.csv
└── validation_5k.csv
```



## Environment Setup
This section configures the experimental environment for fine-tuning and evaluating a BERT-based text classification model. It defines file paths for the cleaned training, validation, and test datasets, sets a fixed random seed for reproducibility, and determines the appropriate computation device depending on whether the script is running in Google Colab or on a local machine (CPU, CUDA, or Apple MPS).

The setup also explicitly disables TensorFlow usage to avoid backend conflicts and ensures that all subsequent training and evaluation steps operate under consistent hardware and configuration assumptions.

## Load Data
This section loads the cleaned CSV datasets for training, validation, and testing into pandas DataFrames. These datasets contain Reddit post titles and corresponding binary labels for fake news classification.

In addition, the code removes rows corresponding to known corrupted indices using precomputed text files. This ensures that all remaining samples are valid and prevents tokenization or training errors caused by malformed data. The result is a clean and reliable dataset for downstream modeling.

## Compute Class Proportions
This section computes the proportion of each class in the training dataset for the binary classification task. Specifically, it calculates the fraction of samples labeled as fake news versus non-fake news.

These proportions provide insight into class imbalance in the data and are later used to construct a loss function that compensates for skewed label distributions.

## Define Prior Adjusted Loss Criterion
This section defines a custom weighted cross-entropy loss function using the previously computed class proportions. The weights are assigned such that underrepresented classes receive higher importance during training.

By incorporating class priors into the loss function, the model is encouraged to learn balanced decision boundaries rather than favoring the majority class.

## Fetch BERT From HuggingFace
This section loads a pretrained BERT-Base, Uncased model and its corresponding tokenizer from Hugging Face. The model is initialized with a classification head configured for binary classification.

Using a pretrained language model allows the system to leverage rich linguistic representations learned from large-scale corpora, providing a strong baseline before fine-tuning on the task-specific dataset.

## Create Hugging Face Datasets
This section converts the pandas DataFrames into Hugging Face `Dataset` objects for training, validation, and testing. These dataset objects integrate seamlessly with the Hugging Face training ecosystem.

This conversion enables efficient tokenization, batching, and metric computation through the Trainer API.

## Tokenize Text Data
This section defines and applies a tokenization function that converts raw text titles into BERT-compatible inputs. Each sample is tokenized with truncation and padding to the model’s maximum sequence length.

The function also constructs attention masks, token type IDs, and explicitly assigns the binary label used for supervised learning. This step transforms raw text into numerical representations suitable for model training and evaluation.

## Define Evaluation Metrics
This section loads standard classification metrics from the Hugging Face `evaluate` library, including accuracy, precision, recall, and F1 score.

These metrics are later used to provide a comprehensive evaluation of model performance, capturing both overall accuracy and class-specific behavior.

## Define compute_metrics Function
This section defines a custom `compute_metrics` function that processes model logits and ground-truth labels to compute multiple evaluation metrics. Predictions are obtained by selecting the class with the highest logit score.

The function reports class-specific precision, recall, and F1 scores, as well as macro, micro, and weighted F1 metrics. This provides a detailed and balanced assessment of model performance across classes.

## Subclass the Trainer to Use Custom Loss
This section defines a subclass of the Hugging Face `Trainer` that overrides the default loss computation. Instead of using the model’s internal loss, it applies the custom class-weighted cross-entropy loss defined earlier.

This customization ensures that class imbalance is explicitly accounted for during training while still leveraging the Trainer’s training loop and evaluation infrastructure.

## Initialize TrainingArguments and Trainer
This section configures the training process by specifying hyperparameters such as batch size, learning rate, number of epochs, checkpointing frequency, and evaluation intervals.

The customized Trainer is then initialized with the model, datasets, training arguments, and metric computation function. This setup defines the full fine-tuning pipeline.

## Evaluate Pretrained Model
This section evaluates the pretrained (non–fine-tuned) BERT model on the training, validation, and test datasets. Evaluation metrics are computed using the previously defined functions.

Results for each split are saved to CSV files, establishing a baseline performance before fine-tuning.

## Train the Model (Fine-Tuning)
This section fine-tunes the pretrained BERT model on the training dataset using the configured Trainer. Training resumes from the latest checkpoint if available to save computation time.

After training completes, the final model weights and trainer state are saved for reproducibility and future use.

## Evaluate Fine-Tuned Model
This final section evaluates the fine-tuned BERT model on the training, validation, and test datasets using the same evaluation procedure as before.

The resulting metrics are saved to CSV files, enabling direct comparison between the pretrained baseline and the fine-tuned model to assess the impact of task-specific training.

