import torch
import torch.nn as nn



class FourierFTAdapter(nn.Module):
    def __init__(self, n_freq, alpha, base_layer, d1, d2):
        super(FourierFTAdapter, self).__init__()
        self.alpha = alpha
        self.base_layer = base_layer
        self.d1 = d1
        self.d2 = d2
        self.n_freq = n_freq
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.spectrum = nn.Parameter(
            torch.randn(n_freq, dtype=torch.complex64) * 0.01, requires_grad=True
        )
        indices = torch.randperm(d1 * d2)[:n_freq]
        self.register_buffer('freq_indices', indices)

    def forward(self, x):
        F_mat = torch.zeros(self.d1 * self.d2, dtype=torch.complex64, device=x.device)
        F_mat[self.freq_indices] = self.spectrum
        F_mat = F_mat.view(self.d1, self.d2)
        Delta_W = torch.fft.ifft2(F_mat).real * self.alpha
        h = self.base_layer(x)
        if x.dim() == 2:
            h = h + torch.matmul(x, Delta_W)
        elif x.dim() == 3:
            h = h + torch.einsum('bnd,df->bnf', x, Delta_W)
        return h


class FourierFTAdapterFixed(nn.Module):
    def __init__(self, n_freq, alpha, base_layer, d1, d2, seed=42):
        super(FourierFTAdapterFixed, self).__init__()
        self.alpha = alpha
        self.base_layer = base_layer
        self.d1 = d1
        self.d2 = d2
        self.n_freq = n_freq
        for p in self.base_layer.parameters():
            p.requires_grad = False
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(d1 * d2, generator=g)[:n_freq]
        self.register_buffer('freq_indices', indices)
        self.spectrum_real = nn.Parameter(torch.randn(n_freq) * 0.01)
        self.spectrum_imag = nn.Parameter(torch.randn(n_freq) * 0.01)

    def forward(self, x):
        spectrum = torch.complex(self.spectrum_real, self.spectrum_imag)
        F_mat = torch.zeros(self.d1 * self.d2, dtype=torch.complex64, device=x.device)
        F_mat[self.freq_indices] = spectrum
        F_mat = F_mat.view(self.d1, self.d2)
        Delta_W = torch.fft.ifft2(F_mat).real * self.alpha
        h = self.base_layer(x)
        if x.dim() == 2:
            h = h + torch.matmul(x, Delta_W)
        elif x.dim() == 3:
            h = h + torch.einsum('bnd,df->bnf', x, Delta_W)
        return h
