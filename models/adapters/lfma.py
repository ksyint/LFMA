import torch
import torch.nn as nn



class LFMAAdapter(nn.Module):
    def __init__(self, alpha, base_layer, delta_W_init, top_k_ratio=0.05):
        super(LFMAAdapter, self).__init__()
        self.alpha = alpha
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        d1, d2 = delta_W_init.shape
        self.d1 = d1
        self.d2 = d2
        W_freq = torch.fft.fft2(delta_W_init)
        magnitude = torch.abs(W_freq)
        k = max(int(d1 * d2 * top_k_ratio), 1)
        _, topk_indices = torch.topk(magnitude.view(-1), k)
        self.register_buffer('mask', torch.zeros((d1, d2), dtype=torch.bool))
        self.mask.view(-1)[topk_indices] = True
        self.c = nn.Parameter(W_freq[self.mask].clone().detach(), requires_grad=True)

    def forward(self, x):
        F_mat = torch.zeros(self.d1, self.d2, dtype=torch.complex64, device=x.device)
        F_mat[self.mask.to(x.device)] = self.c
        Delta_W = torch.fft.ifft2(F_mat).real * self.alpha
        h = self.base_layer(x)
        if x.dim() == 2:
            h = h + torch.matmul(x, Delta_W)
        elif x.dim() == 3:
            h = h + torch.einsum('bnd,df->bnf', x, Delta_W)
        return h


class LFMAAdapterConv(nn.Module):
    def __init__(self, alpha, base_layer, top_k_ratio=0.05):
        super(LFMAAdapterConv, self).__init__()
        self.alpha = alpha
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        weight = base_layer.weight.data
        self.weight_shape = weight.shape
        flat_weight = weight.view(weight.shape[0], -1)
        delta_W_init = flat_weight * 0.01
        d1, d2 = delta_W_init.shape
        self.d1 = d1
        self.d2 = d2
        W_freq = torch.fft.fft2(delta_W_init)
        magnitude = torch.abs(W_freq)
        k = max(int(d1 * d2 * top_k_ratio), 1)
        _, topk_indices = torch.topk(magnitude.view(-1), k)
        self.register_buffer('mask', torch.zeros((d1, d2), dtype=torch.bool))
        self.mask.view(-1)[topk_indices] = True
        self.c = nn.Parameter(W_freq[self.mask].clone().detach(), requires_grad=True)

    def forward(self, x):
        F_mat = torch.zeros(self.d1, self.d2, dtype=torch.complex64, device=x.device)
        F_mat[self.mask.to(x.device)] = self.c
        Delta_W = torch.fft.ifft2(F_mat).real * self.alpha
        h = self.base_layer(x)
        return h
