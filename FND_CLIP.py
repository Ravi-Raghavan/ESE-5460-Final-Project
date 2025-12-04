## Ensure TensorFlow is not used
import os
os.environ["USE_TF"] = "0"

# Import necessary software
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from transformers import CLIPModel, BertModel, BertTokenizer, CLIPProcessor, CLIPTokenizer

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
        x = torch.cat([m_txt, m_img, m_multi], dim=1)  # (B, L, 3)

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
        classifier_hidden=256,
        dropout=0.2
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

        # 3. Setup Multimodal (Text + Image) Encoder
        self.multimodal_encoder = CLIPModel.from_pretrained(clip_model_name)
        self.multimodal_encoder_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    # txt(B, ), List of Text Strings
    # img(B, C_in = 3, H_in = 224, W_in = 224), List of Corresponding Imagess
    def forward(self, txt, img):
        # Compute ResNet Image Features
        image_features = self.image_encoder(img) # Output Shape: (B, 2048)
        print(f"Shape of Image Features: {image_features.shape}")

        # Compute BERT Text Features
        encoding = self.text_encoder_tokenizer(
            txt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ) # Tokenize text

        text_outputs = self.text_encoder(**encoding) # Compute BERT Output
        text_features = text_outputs.last_hidden_state[:, 0, :] # Use [CLS] token as text feature
        print(f"Shape of Text Features: {text_features.shape}") # Output Shape: (B, 768)

        # Compute CLIP Text and Image Features
        encoding = self.multimodal_encoder_tokenizer(
            txt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(img.device) # Tokenize text

        multimodal_image_features = self.multimodal_encoder.get_image_features(img) # Compute Image Features
        multimodal_text_features = self.multimodal_encoder.get_text_features(**encoding) # Compute Text Features
        print(f"Shape of Multimodal Image Features: {multimodal_image_features.shape}") # Output Shape: (B, 512)
        print(f"Shape of Multimodal Text Features: {multimodal_text_features.shape}") # Output Shape: (B, 512)

# Run Smoke Tests
if __name__ == "__main__":
    # Instantiate Model
    model = FND_CLIP()
    model.eval()

    # Sample forward pass
    B = 2
    text_samples = ["a", "b"]
    image_samples = torch.randn(B, 3, 224, 224)
    model(text_samples, image_samples)