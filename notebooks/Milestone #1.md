## Environment Setup
This section establishes the overall configuration required for the data processing pipeline. It defines the directory structure used to store raw and processed data, specifies file paths for the training, validation, and test splits, and sets a fixed random seed to ensure reproducibility. By centralizing these settings at the top of the script, the pipeline remains consistent and easy to modify.

The environment setup also ensures that all subsequent operations reference the correct data locations and follow the same experimental conditions, which is critical for reproducible multimodal experiments.

## Load Data
This section loads the training, validation, and test datasets from tab-separated value (TSV) files into pandas DataFrames. Each split represents a predefined partition of the multimodal Fakeddit dataset and contains text, metadata, image URLs, and multiple label formats.

After loading, the code inspects the datasets using summary and preview commands to verify that the files were read correctly, confirm the presence of expected columns, and identify missing values or inconsistencies that must be addressed before further processing.

## Feature Selection and Rationale
This section narrows the dataset to a carefully chosen subset of features that are relevant for multimodal fake news detection. Columns that are identifiers, redundant, or unlikely to contribute meaningful predictive signal are removed, while features capturing textual content, temporal context, engagement metrics, community information, and image access are retained.

The selected feature set balances modeling relevance with practicality, ensuring that all retained columns either serve as model inputs, support downstream analysis, or represent target labels. Applying the same feature selection across all dataset splits guarantees schema consistency.

## Sanity Checks
This section enforces data quality requirements specific to multimodal learning. Samples missing either textual content or image URLs are removed to ensure that each example contains both modalities. Entries without associated images are filtered out, and samples with missing labels are discarded to maintain valid supervision.

These checks also simplify downstream processing by guaranteeing that every remaining sample has complete text, image, and label information. Resetting indices ensures clean, contiguous DataFrames after filtering.

## Convert Data Types
This section standardizes column data types to ensure consistency across the dataset and avoid downstream errors. Text-based fields are explicitly cast as strings, while Unix timestamps are converted into datetime objects to enable temporal analysis.

By enforcing consistent data types across all splits, this step ensures that subsequent preprocessing, feature engineering, and modeling steps can operate reliably without additional type handling.
