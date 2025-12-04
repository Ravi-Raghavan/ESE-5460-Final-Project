## Ensure TensorFlow is not used
import os
os.environ["USE_TF"] = "0"

# Import necessary software
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from transformers import CLIPModel, BertModel, BertTokenizer, CLIPTokenizer

# Use CPU/MPS if possible
import sys
device = None
if "google.colab" in sys.modules:
    # Running in Colab
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    # Not in Colab (e.g., Mac)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print("Using device:", device)

# Projection Head: 2-layer MLP (Reference: Page 4, Figure 2 of Paper)
class ProjectionHead(nn.Module):
    # in_dim: Number of input features to the Projection Head
    # hidden_dim: Size of hidden state representation
    # out_dim: Size of output dimension
    # dropout: dropout rate
    def __init__(self, in_dim, hidden_dim=256, out_dim=64, dropout=0.2):
        super().__init__()

        # Sequence 1: FC -> BN -> ReLU
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # Define Dropout 
        self.dropout = nn.Dropout(dropout)

        # Sequence 2: FC -> BN -> ReLU
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    # Input Shape: (B, D) where B is batch size and D is in_dim
    def forward(self, x):
        # Sequence 1: FC -> BN -> ReLU
        x = self.fc1(x) # Shape: B x hidden_dim
        x = self.bn1(x) # Shape: B x hidden_dim
        x = self.relu(x) # Shape: B x hidden_dim

        # Dropout
        x = self.dropout(x) # Shape: B x hidden_dim

        # Sequence 2: FC -> BN -> ReLU
        x = self.fc2(x) # Shape: B x out_dim
        x = self.bn2(x) # Shape: B x out_dim
        x = self.relu(x) # Shape: B x out_dim

        # Return Output
        return x # Shape: B x out_dim

# Modality-Wise Attention
class ModalityWiseAttention(nn.Module):
    # feat_dim = L (feature length) of each modality
    def __init__(self, feat_dim):
        super().__init__()

        # Store feature dimension
        self.feat_dim = feat_dim
        
        # Define the MLP Components
        self.fc1 = nn.Linear(3, 3)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(3, 3)
        self.sigmoid = nn.Sigmoid()

    # m_txt: Text Features, Shape: B (batch size) x L (number of features)
    # m_img: Image Features, Shape: B (batch size) x L (number of features)
    # m_multi: Multimodal Features, Shape: B (batch size) x L (number of features)
    def forward(self, m_txt, m_img, m_multi):
        # Use unsqueeze to change shapes
        m_txt = torch.unsqueeze(m_txt, dim = -1) # Shape: B (batch size) x L (number of features) x 1
        m_img = torch.unsqueeze(m_img, dim = -1) # Shape: B (batch size) x L (number of features) x 1
        m_multi = torch.unsqueeze(m_multi, dim = -1) # Shape: B (batch size) x L (number of features) x 1

        # Concatenate all modalities
        x = torch.cat([m_txt, m_img, m_multi], dim = -1)  # (B, L, 3)

        # Global average pooling
        global_avg_pool = torch.mean(x, dim = 1) # Shape: (B, 3)

        # Global max pooling
        global_max_pool, _ = torch.max(x, dim = 1) # Shape: (B, 3)

        # Summation of pooled vectors to get initial attention weights
        x = global_avg_pool + global_max_pool # Shape: (B, 3)

        # Pass through MLP w/ GeLU Activation
        x = self.gelu(self.fc1(x)) # Shape: (B, 3)
        x = self.gelu(self.fc2(x)) # Shape: (B, 3)
        x = self.sigmoid(x) # Shape: (B, 3)

        # Get final attention weights
        w_txt = x[:, 0].unsqueeze(1).unsqueeze(2) # Shape: (B, 1, 1)
        w_img = x[:, 1].unsqueeze(1).unsqueeze(2) # Shape: (B, 1, 1)
        w_multi = x[:, 2].unsqueeze(1).unsqueeze(2) # Shape: (B, 1, 1)

        # sum along modality to get aggregated feature
        mAgg = w_txt * m_txt + w_img * m_img + w_multi * m_multi  # (B, L, 1)
        mAgg = torch.squeeze(mAgg, dim = -1) # Shape: (B, L)
        return mAgg

# Define Final Classification Head
class ClassificationHead(nn.Module):
    # in_dim: Number of input features to the Classification Head
    # hidden_dim: Size of hidden state representation
    # out_dim: Size of output dimension
    def __init__(self, in_dim, hidden_dim = 64, out_dim = 2):
        super().__init__()

        # Sequence 1: FC -> ReLU
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()

        # Sequence 2: FC
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    
    # Shape of x: B(batch size) x d(# of features)
    def forward(self, x):
        # Pass through first layer
        x = self.relu(self.fc1(x))

        # Pass through second layer
        x = self.fc2(x)

        # Return output
        return x

# Fake News Detection(FND) CLIP Model
class FND_CLIP(nn.Module):
    # resnet_model_name: Name of resnet model
    # clip_model_name: Name of CLIP model
    # bert_model_name: Name of BERT Model
    def __init__(
        self,
        resnet_model_name = "resnet101",
        clip_model_name='openai/clip-vit-base-patch32',
        bert_model_name='bert-base-uncased',
        proj_hidden=256,
        proj_out=64,
        classifier_hidden=64,
        dropout=0.2,
        momentum=0.1
    ):
        super().__init__()   

        # Sanity Check
        assert resnet_model_name == "resnet101"

        # 1. Setup ResNet Image Encoder
        # Replace the final fully connected layer with Identity because we only need the ResNet feature embeddings.
        self.image_encoder = tv_models.resnet101(weights='IMAGENET1K_V1')
        self.image_encoder.fc = nn.Identity()

        # 2. Setup BERT Text Encoder
        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        self.text_encoder_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Freeze BERT weights
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 3. Setup Multimodal (Text + Image) Encoder
        self.multimodal_encoder = CLIPModel.from_pretrained(clip_model_name)
        self.multimodal_encoder_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # Freeze CLIP weights
        for param in self.multimodal_encoder.parameters():
            param.requires_grad = False

        # 4. Set up Text Projection Head
        self.pTxt = ProjectionHead(in_dim = 1280, hidden_dim = proj_hidden, out_dim = proj_out, dropout = dropout)
        self.pImg = ProjectionHead(in_dim = 2560, hidden_dim = proj_hidden, out_dim = proj_out, dropout = dropout)
        self.pMix = ProjectionHead(in_dim = 1024, hidden_dim = proj_hidden, out_dim = proj_out, dropout = dropout)

        # 5. Set up Modality-Wise Attention
        self.attention = ModalityWiseAttention(feat_dim = proj_out)

        # 6. Set up Final Classification Head
        self.classification_head = ClassificationHead(in_dim = proj_out, hidden_dim = classifier_hidden, out_dim = 2)

        # Set up Running Buffers
        self.momentum = momentum
        self.eps = 1e-8
        self.register_buffer("running_mean", torch.tensor(0.0, device=device))
        self.register_buffer("running_var", torch.tensor(1.0, device=device))
    
    # Shape: fCLIP_T (B, 512)
    # Shape: fCLIP_I (B, 512)
    def compute_multimodal_features(self, fCLIP_T, fCLIP_I):
        sim = F.cosine_similarity(fCLIP_T, fCLIP_I) # Compute cosine similarity, Shape: (B, )
        fMix = torch.cat((fCLIP_T, fCLIP_I), dim = 1) # Shape: (B, 512 + 512 = 1024)
        
        if self.training:
            batch_mean = sim.mean() # Mean
            batch_var = sim.var() # Variance

            # update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean, var = batch_mean, batch_var
        else:
            # use running stats for eval
            mean, var = self.running_mean, self.running_var
        
        # standardize similarity
        sim_std = (sim - mean) / torch.sqrt(var + self.eps)

        # weight multimodal features
        sim_weight = torch.sigmoid(sim_std).unsqueeze(1) # Shape: (B, 1)

        mMix = sim_weight * self.pMix(fMix) # Shape: (B, 64)

        # Return fMix and mMix
        return fMix, mMix

    # txt(B, ), List of Text Strings
    # img(B, C_in = 3, H_in = 224, W_in = 224), List of Corresponding Imagess
    def forward(self, txt, img):
        # Compute BERT Text Features
        text_encoding = self.text_encoder_tokenizer(
            txt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device) # Tokenize text

        fBERT = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :] # Use [CLS] token as text feature
        # Shape: (B, 768)

        # Compute ResNet Image Features
        fResNet = self.image_encoder(img) # Output Shape: (B, 2048)

        # Compute CLIP Text and Image Features
        text_encoding = self.multimodal_encoder_tokenizer(
            txt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device) # Tokenize text

        fCLIP_T = self.multimodal_encoder.get_text_features(**text_encoding) # Compute CLIP Text Features
        fCLIP_I = self.multimodal_encoder.get_image_features(img) # Compute CLIP Image Features

        # Concatenate 
        fTxt = torch.cat((fBERT, fCLIP_T), dim = 1) # Shape: (B, 768 + 512 = 1280)
        fImg = torch.cat((fResNet, fCLIP_I), dim = 1) # Shape: (B, 2048 + 512 = 2560)

        # Compute mTxt and mImg
        mTxt = self.pTxt(fTxt) # Shape: (B, 64)
        mImg = self.pImg(fImg) # Shape: (B, 64)

        fMix, mMix = self.compute_multimodal_features(fCLIP_T, fCLIP_I) 
        # fMix Shape: (B, 512 + 512 = 1024)
        # mMix Shape: (B, 64)

        # Perform Modality-Wise Attention
        mAgg = self.attention(mTxt, mImg, mMix) # Shape: (B, 64)

        # Compute Final Logits
        logits = self.classification_head(mAgg) # Shape: (B, 2)
        return logits

# Run Smoke Tests
if __name__ == "__main__":
    # Instantiate Model
    model = FND_CLIP().to(device)

    # Sample Data
    B = 10
    text_samples = ["a" for _ in range(B)]
    image_samples = torch.randn(B, 3, 224, 224).to(device)
    ground_truth = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], device = device)

    # Sample forward pass
    print(f"Training Forward Pass")
    model.train()
    model(text_samples, image_samples)

    # Divider
    print("------------------------------------------------")

    # Sample eval forward pass 
    print(f"Eval Forward Pass")
    model.eval()
    model(text_samples, image_samples)

    # Divider
    print("------------------------------------------------")

    # Sample backward pass
    print(f"Backward Pass")
    criterion = nn.CrossEntropyLoss()
    logits = model(text_samples, image_samples)
    loss = criterion(logits, ground_truth)
    loss.backward()

    ## If all of these run successfully, ready to implement training loop and training infrastructure