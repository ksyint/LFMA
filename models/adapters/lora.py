import torch
import torch.nn as nn
import math



class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0, dropout=0.0):
        super(LoRALinear, self).__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.base_layer(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return h + lora_out


class LoRAEmbedding(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0):
        super(LoRAEmbedding, self).__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        num_embeddings = base_layer.num_embeddings
        embedding_dim = base_layer.embedding_dim
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(num_embeddings, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, embedding_dim))
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        h = self.base_layer(x)
        after_A = nn.functional.embedding(x, self.lora_A)
        return h + (after_A @ self.lora_B) * self.scaling
