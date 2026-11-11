"""NOCADILAC (Nonlinear Causal Discovery with Latent Confounders) baseline.

Reference: Kaltenpoth, D., & Vreeken, J. (2023). Nonlinear causal
discovery with latent confounders. ICML.

We reuse the original VAE formulation. The latent structure is read off the
encoder's inferred Z: Y is clustered to its strongest correlated latent,
and A (X -> Z) is recovered by ridge regression of Z on X.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NOCADILACEncoder(nn.Module):
    """Inference network q(Z | Y)."""

    def __init__(self, dy, dz, hidden_dims=(128, 64)):
        super().__init__()
        self.dy = dy
        self.dz = dz

        layers = []
        prev_dim = dy
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev_dim, dz)
        self.logvar_layer = nn.Linear(prev_dim, dz)

    def forward(self, y):
        h = self.encoder(y)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class NOCADILACDecoder(nn.Module):
    """Generative network p(Y | Z)."""

    def __init__(self, dz, dy, hidden_dims=(128, 64)):
        super().__init__()
        self.dz = dz
        self.dy = dy

        layers = []
        prev_dim = dz
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, dy)
        self.logvar_layer = nn.Linear(prev_dim, dy)

    def forward(self, z):
        h = self.decoder(z)
        mu = self.output_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class NOCADILACModel(nn.Module):
    def __init__(self, dy, dz, hidden_dims=(128, 64)):
        super().__init__()
        self.dy = dy
        self.dz = dz

        self.encoder = NOCADILACEncoder(dy, dz, hidden_dims)
        self.decoder = NOCADILACDecoder(dz, dy, hidden_dims)

        self.causal_mask = nn.Parameter(torch.ones(dy, dz))

    def encode(self, y):
        return self.encoder(y)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y):
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z)
        return recon_mu, recon_logvar, mu, logvar, z


def nocadilac_loss_function(recon_mu, recon_logvar, y, mu, logvar, z, causal_mask, beta=1.0):
    """Reconstruction NLL plus a beta-weighted KL term."""
    recon_loss = nn.functional.gaussian_nll_loss(
        recon_mu, y, torch.exp(recon_logvar), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_nocadilac(data, dz, hidden_dims=(128, 64), epochs=200, lr=1e-3,
                    batch_size=64, beta=1.0, seed=42, device='cpu'):
    """Train a NOCADILAC model and return (model, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dy = data.shape[1]
    model = NOCADILACModel(dy, dz, hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data_tensor = torch.FloatTensor(data).to(device)
    n_samples = data.shape[0]

    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]
            batch = data_tensor[batch_idx]

            recon_mu, recon_logvar, mu, logvar, z = model(batch)
            loss, recon_l, kl_l = nocadilac_loss_function(
                recon_mu, recon_logvar, batch, mu, logvar, z,
                model.causal_mask, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()
            n_batches += 1

        history['loss'].append(total_loss / n_batches)
        history['recon_loss'].append(total_recon / n_batches)
        history['kl_loss'].append(total_kl / n_batches)

    return model, history


def extract_causal_structure(model, data, threshold=0.2):
    """Read the B matrix (Z -> Y) from the correlation of Y with inferred Z."""
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data)
        mu, logvar = model.encode(data_tensor)
        z = mu.numpy()

    dz = z.shape[1]
    dy = data.shape[1]

    z_std = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)
    y_std = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    corr_matrix = np.abs(y_std.T @ z_std / data.shape[0])

    B_pred = corr_matrix.T

    cluster_labels = np.argmax(corr_matrix, axis=1)

    B_binary = np.zeros((dz, dy))
    for j in range(dy):
        B_binary[cluster_labels[j], j] = 1.0

    return B_binary, B_pred, cluster_labels


def run_nocadilac(X, Y, dz=None, hidden_dims=(128, 64), epochs=200,
                  lr=1e-3, batch_size=64, seed=42, device='cpu', **kwargs):
    """Run NOCADILAC and return a result dict compatible with compute_metrics."""
    data = Y
    dy = data.shape[1]
    dx = X.shape[1] if X is not None else 0

    if dz is None:
        dz = max(2, dy // 5)

    model, history = train_nocadilac(
        data, dz, hidden_dims=hidden_dims, epochs=epochs,
        lr=lr, batch_size=batch_size, seed=seed, device=device)

    B_pred, B_raw, cluster_labels = extract_causal_structure(model, data, threshold=0.2)

    # Recover A (X -> Z) by ridge-regressing the inferred Z on X.
    model.eval()
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data).to(device)
        mu, logvar = model.encode(data_tensor)
        Z_inferred = mu.cpu().numpy()

    A_pred = np.zeros((dx, dz))
    if X is not None and dx > 0:
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        Z_std = (Z_inferred - Z_inferred.mean(axis=0)) / (Z_inferred.std(axis=0) + 1e-8)

        try:
            XtX = X_std.T @ X_std + 1e-6 * np.eye(dx)
            XtZ = X_std.T @ Z_std
            A_regression = np.linalg.solve(XtX, XtZ)
            A_pred = (np.abs(A_regression) > 0.2).astype(float)
        except np.linalg.LinAlgError:
            A_pred = np.zeros((dx, dz))

    return {
        'A_pred': A_pred,
        'B_pred': B_pred,
        'B_raw': B_raw,
        'dz_pred': dz,
        'cluster_labels': cluster_labels,
        'history': history,
        'model': model,
        'method': 'NOCADILAC',
    }
