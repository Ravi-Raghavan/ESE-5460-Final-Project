# Multimodal Misinformation Detection on Social Media [Project ID: 63]

**ESE 5460: Principles of Deep Learning**

This repository contains our implementation of multimodal misinformation detection models that jointly analyze text and images from social media posts. We evaluate unimodal and multimodal baselines on the **Fakeddit** dataset and explore **CLIP-guided multimodal fusion** to improve fake news detection performance.

## Authors
- Ravi Raghavan (rr1133@seas.upenn.edu)  
- Dhruv Verma (vdhruv@seas.upenn.edu)  
- Raafae Zaki (rzaki2@seas.upenn.edu)  

## Dataset
We use the **Fakeddit** dataset, which provides labeled image–text pairs for fine-grained fake news classification.

- Text: post titles and descriptions  
- Images: associated social media images  
- Labels: misinformation categories  

> **Note to Instructor:** Dataset details follow the setup described in *r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection*

> **Note to Instructor:** Due to size constraints, the dataset is not included in this repository.

## Dataset Preprocessing
Following the instructions from the [r/Fakeddit paper](https://arxiv.org/pdf/1911.03854), we obtained the dataset from the official [Fakeddit GitHub repository](https://github.com/entitize/fakeddit).  

The repository provides a link to the dataset’s [Google Drive Folder](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing)

Since our project focuses on **multimodal analysis**, we use only the multimodal samples, which contain **both text and images**.

Specifically, we downloaded the following files from the above Google Drive Link
- `multimodal_train.tsv`  
- `multimodal_validate.tsv`  
- `multimodal_test_public.tsv`  



## Models Implemented

### 1. Text Unimodal Baseline
- Pretrained **BERT (bert-base-uncased)**
- Classification head on top of [CLS] embedding
- Comparison of pretrained vs. fine-tuned BERT

### 2. Image Unimodal Baseline
- Pretrained **ResNet-101**
- Image features were taken prior to last FC Layer
- Comparison of pretrained vs. fine-tuned ResNet

### 3. Multimodal Baseline
- Concatenation of BERT text embeddings and ResNet image features
- Joint classifier trained end-to-end

### 4. CLIP-Guided Multimodal Model
- CLIP-based joint text–image representations
- Fusion guided by CLIP embeddings
- Ablation Study between different attention mechanisms:
  - QKV (self-attention)
  - Modality-aware attention

