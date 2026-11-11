"""Training loop for LMCM-VAE, including a simple dz selection helper."""

import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lmcm.model import LMCMVAE, compute_loss


def train_lmcm(X, Y, dz,
               hidden_dims=None,
               B_param='unconstrained',
               epochs=1000,
               lr=1e-3,
               batch_size=64,
               lambda_A=0.01,
               lambda_B=0.01,
               device='cpu',
               verbose=True,
               print_every=100,
               seed=None):
    """Train an LMCM-VAE and return the fitted model and its training history.

    The best model (lowest averaged loss) is restored before returning.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    X = torch.FloatTensor(X).to(device)
    Y = torch.FloatTensor(Y).to(device)

    dx = X.shape[1]
    dy = Y.shape[1]
    n_samples = X.shape[0]

    model = LMCMVAE(dx, dy, dz, hidden_dims, B_param=B_param).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    history = {
        'loss': [],
        'elbo': [],
        'recon_loss': [],
        'kl_loss': [],
    }

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        epoch_loss = 0.0
        epoch_elbo = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            loss, elbo, recon_loss, kl_loss = compute_loss(
                model, X_batch, Y_batch, lambda_A, lambda_B
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_elbo += elbo.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_elbo = epoch_elbo / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        history['loss'].append(avg_loss)
        history['elbo'].append(avg_elbo)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Loss: {avg_loss:.4f} - "
                  f"ELBO: {avg_elbo:.4f} - "
                  f"Recon: {avg_recon:.4f} - "
                  f"KL: {avg_kl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def select_dz(X, Y, dz_range, device='cpu', **train_kwargs):
    """Train one model per candidate dz and pick the one with the best ELBO."""
    results = {}
    best_elbo = -float('inf')
    best_dz = None

    for dz in dz_range:
        model, history = train_lmcm(X, Y, dz, device=device, **train_kwargs)
        final_elbo = history['elbo'][-1]
        results[dz] = {
            'elbo': final_elbo,
            'model': model,
            'history': history,
        }

        if final_elbo > best_elbo:
            best_elbo = final_elbo
            best_dz = dz

    return best_dz, results
