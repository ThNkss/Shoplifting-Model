import torch.nn as nn
import torch
import torch.nn.functional as F

class VideoClassifier(nn.Module):
    def __init__(self, cnn, rnn, num_classes=2):
        super().__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.feat_norm = nn.LayerNorm(1280)  # Normalize CNN features
        self.attention = nn.Sequential(      # Temporal attention
            nn.Linear(rnn.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(rnn.hidden_dim * 2, num_classes)
        self.locator = nn.Linear(rnn.hidden_dim * 2, 1)  # For temporal localization

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
            
    def forward(self, x, lengths):
        # CNN Feature Extraction
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        cnn_features = self.feat_norm(self.cnn(x)).view(b, t, -1) # Pass through CNN
        cnn_features = cnn_features.view(b, t, -1)  # Reshape back for RNN
        
        # RNN Processing
        rnn_out = self.rnn(cnn_features, lengths)  # [b, t, hidden*2]
        
        # Masked Attention Pooling
        mask = torch.arange(t, device=x.device)[None,:] < lengths[:,None]
        attn_weights = self.attention(rnn_out).squeeze(-1)  # [b, t]
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled_features = (rnn_out * attn_weights.unsqueeze(-1)).sum(1)
        
        # Multi-Task Output
        classification = self.classifier(pooled_features)
        localization = self.locator(rnn_out)  # Frame-level scores
        
        return classification, localization