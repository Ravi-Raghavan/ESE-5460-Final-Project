# Notebook Description

To further reduce computational overhead during training, we performed additional downsampling of the validation and test splits. Although the initial subsampled splits contained approximately 33,000 examples each, evaluating the model at frequent checkpoints—every 50–100 weight updates—proved to be prohibitively expensive.  

To address this, we **randomly downsampled the validation and test sets to 5,000 samples each**, while ensuring that the original label distributions were preserved. This stratified downsampling allowed for efficient evaluation without distorting class proportions.  

By reducing the size of these splits, we were able to maintain frequent model checkpointing and monitoring of validation performance, while keeping the training loop tractable. This approach balances evaluation efficiency with representativeness during model development.