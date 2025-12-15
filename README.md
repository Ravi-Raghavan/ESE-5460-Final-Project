# Multimodal Misinformation Detection on Social Media [Project ID: 63]

**ESE 5460: Principles of Deep Learning**

This repository contains our implementation of multimodal misinformation detection models that jointly analyze text and images from social media posts. We evaluate unimodal and multimodal baselines on the **Fakeddit** dataset and explore **CLIP-guided multimodal fusion** to improve fake news detection performance.

## Authors
- Ravi Raghavan (rr1133@seas.upenn.edu)  
- Dhruv Verma (vdhruv@seas.upenn.edu)  
- Raafae Zaki (rzaki2@seas.upenn.edu)  

## Dataset
We use the **Fakeddit** dataset, a large-scale multimodal dataset collected from **Reddit**, designed for fine-grained fake news and misinformation classification.

- **Text**: The text associated with each Reddit post  
- **Images**: Images attached to the corresponding Reddit post  
- **Labels**: Fine-grained misinformation categories indicating the credibility and intent of the post

> **Note to Instructor:** Dataset details follow the setup described in *r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection*

> **Note to Instructor:** Due to file size limitations, the dataset is not included in this repository. Instead, we provide the instructions below for accessing the data directly from the original Google Drive source.

Following the instructions from the [r/Fakeddit paper](https://arxiv.org/pdf/1911.03854), we obtained the dataset from the official [Fakeddit GitHub repository](https://github.com/entitize/fakeddit).  

The repository provides a link to the dataset’s [Google Drive Folder](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing)

Since our project focuses on **multimodal analysis**, we use only the multimodal samples, which contain **both text and images**.

Specifically, we downloaded the following files from the Google Drive link above and stored them locally in a folder named `data/`:
- `multimodal_train.tsv`
- `multimodal_validate.tsv`
- `multimodal_test_public.tsv`

Local File Structure
```text
data/
├── multimodal_train.tsv
├── multimodal_validate.tsv
└── multimodal_test_public.tsv
```

> **Note to Instructor:** We considered the different label schemes provided in the Fakeddit dataset: **`2_way_label`**, **`3_way_label`**, and **`6_way_label`**. Since our primary goal is **binary classification**, we used **`2_way_label`** for all experiments. In our experiments, a label of **0** corresponds to **fake news**, while a label of **1** corresponds to **non-fake news**. The other label schemes (`3_way_label` and `6_way_label`) provide finer-grained categorizations of misinformation, but they were not used in our models to maintain focus on the binary detection task.

## Data Preprocessing

This section describes the preprocessing steps applied to the multimodal Fakeddit dataset prior to modeling and analysis.

First, the training, validation, and test splits were loaded from TSV files into pandas DataFrames. Basic inspections (e.g., previews and summary statistics) were performed to verify successful loading, confirm schema consistency, and identify missing or malformed entries.

Next, we performed **feature selection** to retain only columns relevant to multimodal misinformation detection. Identifier fields, redundant metadata, and features unlikely to contribute meaningful predictive signal were removed. We retained textual content, image-related fields, engagement and contextual metadata, and the target labels. The same feature subset was applied consistently across all dataset splits to ensure a uniform schema.

We then applied **sanity checks** tailored to multimodal learning. Samples missing either textual content or image URLs were removed, ensuring that every example contained both modalities. Entries with missing or invalid labels were also discarded to maintain valid supervision. After filtering, indices were reset to keep the DataFrames clean and contiguous.

We then **converted data types** to ensure consistency across the dataset and avoid downstream errors. Text-based fields were explicitly cast to strings, while Unix timestamps were converted into datetime objects to enable temporal analysis. Enforcing consistent data types across all splits ensures that subsequent preprocessing, feature engineering, and modeling steps can operate reliably without additional type handling.

Because images in the Fakeddit dataset are provided as **URLs rather than raw image files**, we crawled the internet to download the corresponding images for each sample. Since some image URLs were no longer accessible or failed to download, we intentionally sampled a larger number of examples per split to ensure sufficient usable data after crawling.

After cleaning, we performed **dataset subsampling** to accommodate compute constraints and potential image retrieval failures. We initially sampled a larger subset from each split while preserving the original label distribution. Specifically, we used:
- **50,000 samples** for training  
- **50,000 samples** for validation  
- **50,000 samples** for testing  

The subsampling procedure was **stratified by the target label**, ensuring that class proportions remained consistent across all splits. This approach ensured that even if a portion of images failed to download, we still retained ample multimodal data for training and evaluation.

Together, these preprocessing steps produce a clean, consistent, and fully multimodal dataset suitable for reproducible experimentation.

Source Code Reference: The code for data preprocessing was implemented in the following Jupyter Notebook
- [`Milestone #1.ipynb`](notebooks/Milestone%20%231.ipynb)

After crawling the internet for the images and removing samples where image retrieval failed, the final dataset sizes for each split were as follows: 
- Train: 33324
- Validation: 33316
- Test: 33519

These final splits maintain the original label distributions and ensure that every example contains both textual and visual modalities. The resulting dataset is clean, consistent, and fully multimodal, providing a robust foundation for downstream modeling and experimentation.

## Further Downsampling

To further reduce computational overhead during training, we performed additional downsampling of the validation and test splits. Although the initial subsampled splits contained approximately 33,000 examples each, evaluating the model at frequent checkpoints—every 50–100 weight updates—proved to be prohibitively expensive.  

To address this, we **randomly downsampled the validation and test sets to 5,000 samples each**, while ensuring that the original label distributions were preserved. This stratified downsampling allowed for efficient evaluation without distorting class proportions.  

By reducing the size of these splits, we were able to maintain frequent model checkpointing and monitoring of validation performance, while keeping the training loop tractable. This approach balances evaluation efficiency with representativeness during model development.

Source Code Reference: 
- [`ValidationTestDownsample.ipynb`](notebooks/ValidationTestDownsample.ipynb)

## Identifying Corrupted Images

After crawling the image URLs, some images could not be opened because they were corrupted. To handle this, we used the notebooks listed below to identify which samples in the dataset were corrupted. In every model we subsequently train, when loading the data, we filter out these corrupted samples to ensure that only valid images are used during training and evaluation.

Source Code Reference: 
- [`ImageCorruptionAnalysis.ipynb`](notebooks/ImageCorruptionAnalysis.ipynb)
- [`ImageCorruptionAnalysis(5k).ipynb`](notebooks/ImageCorruptionAnalysis(5k).ipynb)

## Models Implemented
Before developing our main CLIP-based models, we began by implementing several baseline models. The purpose of these baselines was to establish reference performance levels and provide a point of comparison for our more complex multimodal models.

### 1. Text Unimodal Baseline
- Utilized the pretrained **BERT (bert-base-uncased)** model to encode **only the text portion of each Reddit post**.  
- Added a **classification head** on top of the [CLS] token embedding to perform binary classification
- Compared **Pretrained BERT** versus **fully fine-tuned BERT**, allowing us to evaluate the benefits of adapting the language model to our specific fake-news detection task.
- Source Jupyter Notebook: [`Milestone #2_Ravi.ipynb`](notebooks/unimodal/text/Milestone%20%232_Ravi.ipynb)

### 2. Image Unimodal Baseline
- Utilized the pretrained **ResNet-101** model to extract visual features from **only the image portion of each Reddit post**.  
- Features were obtained from the layer immediately **before the final fully connected (FC) classification layer**, capturing high-level image representations.  
- Conducted experiments comparing **Pretrained ResNet** versus **Fine-tuned ResNet**, allowing us to assess the benefits of adapting the visual model to our specific fake-news detection task.
- Source Jupyter Notebook: [`Milestone2_Dhruv.ipynb`](notebooks/unimodal/images/Milestone2_Dhruv.ipynb)

### 3. Multimodal Baseline
- Concatenation of BERT text embeddings and ResNet image features
- Joint classifier trained end-to-end

### 4. CLIP-Guided Multimodal Model
- CLIP-based joint text–image representations
- Fusion guided by CLIP embeddings
- Ablation Study between different attention mechanisms:
  - QKV (self-attention)
  - Modality-aware attention