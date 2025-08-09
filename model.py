import math

import torch.nn as nn
import torch
from torch import Tensor


class SLUTransformer(nn.Module):
    def __init__(self, input_dim=320, d_model=128, nhead=4, num_layers=5, num_classes=31):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):              # x: (B, T, 320)
        x = self.input_fc(x)           # (B, T, d_model)
        x = x.permute(1, 0, 2)         # (T, B, d_model)
        x = self.pos_encoder(x)        # (T, B, d_model)
        x = self.encoder(x)            # (T, B, d_model)
        x = x.mean(dim=0)              # (B, d_model)
        return self.classifier(x)      # (B, num_classes)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    model = SLUTransformer(num_classes=31)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
