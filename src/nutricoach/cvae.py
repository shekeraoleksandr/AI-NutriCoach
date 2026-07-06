"""Conditional Variational Autoencoder for nutrient-profile generation (Phase 1, Task 3).

Improves on a plain VAE (the reference project's 'future work') by conditioning on the
target Nutri-Score grade, so we can ask for e.g. a healthy grade-'a' profile on demand.

Public tool interface (used by the Gradio app and, in Phase 2, the RAG agent):
    generate_profile(target_grade="a", n=1) -> list[dict]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

N_FEATURES = len(C.NUTRIENT_FEATURES)
N_CONDITIONS = len(C.NUTRISCORE_GRADES)


class CVAE(nn.Module):
    """Small conditional VAE. Condition = one-hot Nutri-Score grade."""

    def __init__(self, n_features: int = N_FEATURES, n_cond: int = N_CONDITIONS,
                 latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder: [features | condition] -> latent
        self.enc = nn.Sequential(
            nn.Linear(n_features + n_cond, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        # Decoder: [latent | condition] -> features
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + n_cond, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=-1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def loss_fn(recon, x, mu, logvar, beta: float = 1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


# --- Persistence -------------------------------------------------------------
CVAE_PATH = C.MODELS_DIR / "cvae.pt"
SCALER_PATH = C.MODELS_DIR / "cvae_scaler.joblib"   # StandardScaler fit on features


def _grade_onehot(grade: str) -> np.ndarray:
    vec = np.zeros(N_CONDITIONS, dtype=np.float32)
    vec[C.NUTRISCORE_GRADES.index(grade.lower())] = 1.0
    return vec


# --- Tool interface ----------------------------------------------------------
_model_cache = None
_scaler_cache = None


def _load():
    global _model_cache, _scaler_cache
    if _model_cache is None:
        import joblib

        if not CVAE_PATH.exists():
            raise FileNotFoundError(
                "cvae.pt not found — train it first (notebooks/03_cvae.ipynb)"
            )
        model = CVAE()
        model.load_state_dict(torch.load(CVAE_PATH, map_location="cpu"))
        model.eval()
        _model_cache = model
        _scaler_cache = joblib.load(SCALER_PATH)
    return _model_cache, _scaler_cache


def generate_profile(target_grade: str = "a", n: int = 1) -> list[dict]:
    """Sample `n` nutrient profiles conditioned on `target_grade`. Returns list of dicts."""
    model, scaler = _load()
    c = torch.tensor(np.tile(_grade_onehot(target_grade), (n, 1)))
    z = torch.randn(n, model.latent_dim)
    with torch.no_grad():
        x = model.decode(z, c).numpy()
    x = scaler.inverse_transform(x)                 # back to real nutrient units
    x = np.clip(x, 0, None)                          # nutrients can't be negative
    return [dict(zip(C.NUTRIENT_FEATURES, row.tolist())) for row in x]
