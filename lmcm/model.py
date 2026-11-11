"""LMCM-VAE model definition: encoder, structured decoder and ELBO loss.

The decoder encodes the two structural assumptions of the model:
    p(Z | X) = N(A^T X, diag(sigma_Z^2))
    p(Y | Z) = N(B^T Z, diag(sigma_Y^2))
where A (dx x dz) and B (dz x dy) are the learnable causal parameter
matrices. B can be parameterised freely ('unconstrained', with an L1
sparsity prior) or with a softmax over a single parent per Y
('single_parent').
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LMCMEncoder(nn.Module):
    """Inference network q(Z | X, Y)."""

    def __init__(self, dx, dy, dz, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = dx + dy
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev_dim, dz)
        self.logvar_layer = nn.Linear(prev_dim, dz)

    def forward(self, X, Y):
        h = torch.cat([X, Y], dim=1)
        h = self.encoder(h)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class LMCMDecoder(nn.Module):
    """Structured generative model p(Z | X) p(Y | Z)."""

    def __init__(self, dx, dy, dz, B_param='unconstrained'):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.B_param = B_param

        self.A = nn.Parameter(torch.randn(dx, dz) * 0.1)

        if B_param == 'single_parent':
            self.B_logits = nn.Parameter(torch.randn(dy, dz) * 0.1)
            self.B_scales = nn.Parameter(torch.ones(dy) * 0.5)
        else:
            self.B = nn.Parameter(torch.randn(dz, dy) * 0.1)

        self.log_sigma_Z = nn.Parameter(torch.zeros(dz))
        self.log_sigma_Y = nn.Parameter(torch.zeros(dy))

    def get_B_matrix(self):
        if self.B_param == 'single_parent':
            B_weights = F.softmax(self.B_logits, dim=1)
            B_scaled = B_weights * self.B_scales.unsqueeze(1)
            return B_scaled.T
        return self.B

    def prior_params(self, X):
        """Mean and log-variance of p(Z | X)."""
        mu_Z = X @ self.A
        logvar_Z = 2 * self.log_sigma_Z.unsqueeze(0)
        return mu_Z, logvar_Z

    def likelihood_params(self, Z):
        """Mean and log-variance of p(Y | Z)."""
        B = self.get_B_matrix()
        mu_Y = Z @ B
        logvar_Y = 2 * self.log_sigma_Y.unsqueeze(0)
        return mu_Y, logvar_Y

    def forward(self, X, Z):
        mu_Y, logvar_Y = self.likelihood_params(Z)
        return mu_Y, logvar_Y


class LMCMVAE(nn.Module):
    """Full latent mediator causal VAE for the X -> Z -> Y chain."""

    def __init__(self, dx, dy, dz, hidden_dims=None, B_param='unconstrained'):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.B_param = B_param

        self.encoder = LMCMEncoder(dx, dy, dz, hidden_dims)
        self.decoder = LMCMDecoder(dx, dy, dz, B_param=B_param)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, X, Y):
        mu, logvar = self.encoder(X, Y)
        Z = self.reparameterize(mu, logvar)
        return Z, mu, logvar

    def decode(self, X, Z):
        mu_Y, logvar_Y = self.decoder(X, Z)
        return mu_Y, logvar_Y

    def forward(self, X, Y):
        Z, q_mu, q_logvar = self.encode(X, Y)
        mu_Y, logvar_Y = self.decode(X, Z)
        p_mu_Z, p_logvar_Z = self.decoder.prior_params(X)
        return mu_Y, logvar_Y, q_mu, q_logvar, p_mu_Z, p_logvar_Z, Z

    def get_A(self):
        return self.decoder.A.detach().cpu().numpy()

    def get_B(self):
        return self.decoder.get_B_matrix().detach().cpu().numpy()


def gaussian_log_likelihood(x, mu, logvar):
    """Log-likelihood of a diagonal Gaussian, summed over features."""
    return -0.5 * torch.sum(
        logvar + (x - mu) ** 2 / torch.exp(logvar) + np.log(2 * np.pi),
        dim=1,
    )


def kl_divergence_diag(q_mu, q_logvar, p_mu, p_logvar):
    """KL divergence between two diagonal Gaussians, summed over features."""
    return 0.5 * torch.sum(
        p_logvar - q_logvar
        + (torch.exp(q_logvar) + (q_mu - p_mu) ** 2) / torch.exp(p_logvar)
        - 1,
        dim=1,
    )


def compute_loss(model, X, Y, lambda_A=0.01, lambda_B=0.01):
    """ELBO reconstruction + KL plus L1 sparsity on A and B.

    Returns (total_loss, mean_elbo, mean_recon, mean_kl).
    """
    mu_Y, logvar_Y, q_mu, q_logvar, p_mu_Z, p_logvar_Z, Z = model(X, Y)

    recon_loss = -gaussian_log_likelihood(Y, mu_Y, logvar_Y)
    kl_loss = kl_divergence_diag(q_mu, q_logvar, p_mu_Z, p_logvar_Z)

    loss = recon_loss.mean() + kl_loss.mean()

    A = model.decoder.A
    B = model.decoder.get_B_matrix()
    sparse_loss = lambda_A * torch.norm(A, p=1) + lambda_B * torch.norm(B, p=1)

    total_loss = loss + sparse_loss

    return total_loss, (-recon_loss - kl_loss).mean(), recon_loss.mean(), kl_loss.mean()
