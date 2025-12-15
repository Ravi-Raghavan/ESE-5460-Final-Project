# Multimodal Misinformation Detection [Project ID: 63]

**ESE 5460: Principles of Deep Learning — Final Project**

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

Dataset details and preprocessing follow the setup described in *r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection*.

> **Note:** Due to size constraints, the dataset is not included in this repository.


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

