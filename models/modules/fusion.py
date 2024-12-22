import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn

class MultiModalFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_proj = nn.Linear(768, 768)
        self.fusion_proj = nn.Linear(1536, 768)
        self.relu = nn.ReLU()

    def forward(self, image_embedding, text_embedding, uncond_embedding):
        mapped_image = self.image_proj(image_embedding)  # (1, 768)
        mapped_image = self.relu(mapped_image)
        expanded_image = mapped_image.unsqueeze(1).repeat(1, 77, 1)  # (1, 77, 768)
        concat_embedding = torch.cat([text_embedding, expanded_image], dim=-1)  # (1, 77, 1536)
        text_embeddings = self.fusion_proj(concat_embedding)  # (1, 77, 768)
        text_embeddings = torch.cat([text_embeddings, uncond_embedding], dim=0)
        
        return text_embeddings