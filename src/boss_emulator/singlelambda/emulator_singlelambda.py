"""Coordinate-query cross-attention emulator: (labels, position) -> flux.

This mimics :class:`~boss_emulator.emulator.SpectraFormer` -- each scalar
label becomes a key/value memory token and stacked cross-attention blocks let a
query read from that label memory -- with one load-bearing change: the query is
built from a **continuous** pixel position rather than a learned per-pixel
embedding. A position ``p`` in ``[-1, 1]`` is lifted by Fourier features and a
small MLP into a ``d_model`` query token; the refined token is read out to a
scalar flux. Because the position enters continuously, the same model evaluates
*any* set of positions -- pass one position to get one pixel, or the whole
``n_pixels`` grid to reconstruct a full spectrum.

The forward is therefore ``model(x, coords)`` (two arguments), unlike the
single-argument ``model(x)`` of the other three emulators; the
:mod:`boss_emulator.singlelambda.training` / ``.evaluate`` helpers supply the
coordinates. Inputs are standardised and outputs standardised, exactly as the
other models; de-standardisation lives with the
:class:`~boss_emulator.data.Dataset`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from boss_emulator.emulator import Activation, _CrossAttnBlock, resolve_activation


def fourier_features(p: jax.Array, n_fourier: int) -> jax.Array:
    """Lift scalar positions to sinusoidal features at octave frequencies.

    Given positions ``p`` (any shape, values in ``[-1, 1]``) and ``n_fourier``
    frequency bands, return an array with a trailing axis of size
    ``2 * n_fourier`` holding ``[sin(f_k p), cos(f_k p)]`` for
    ``f_k = pi * 2**k``, ``k = 0 .. n_fourier-1``. The lowest band captures the
    global slope across the spectrum; the highest resolves the finest pixel
    spacing. Frequencies are recomputed here (no persisted state), so only
    ``n_fourier`` need be stored to reproduce the encoding.
    """
    p = jnp.asarray(p).reshape(-1)                       # (K,)
    freqs = (2.0 ** jnp.arange(n_fourier)) * jnp.pi      # (F,)
    ang = p[:, None] * freqs[None, :]                    # (K, F)
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)  # (K, 2F)


class SpectraFormerSingleLambda(nnx.Module):
    """Cross-attention decoder with a continuous position query.

    The ``n_features`` scalar labels are projected to ``d_model`` and given a
    learned per-label identity embedding, forming a small key/value memory (as
    in :class:`~boss_emulator.emulator.SpectraFormer`). Each queried position is
    encoded with :func:`fourier_features` and a two-layer MLP into a ``d_model``
    query token; ``n_blocks`` cross-attention blocks let the query read from the
    label memory (residual + pre-LayerNorm), and a linear head maps the refined
    query to its scalar flux.

    Parameters
    ----------
    n_features : int
        Number of input labels (number of memory tokens).
    n_pixels : int
        Length of the full spectrum grid (only stored for bookkeeping; the
        model itself is agnostic to how many positions are queried at once).
    d_model : int
        Token/embedding width shared by queries and memory.
    n_heads : int
        Number of attention heads (must divide ``d_model``).
    n_blocks : int
        Number of stacked cross-attention blocks.
    mlp_dim : int
        Hidden width of each block's feed-forward network.
    n_fourier : int
        Number of Fourier frequency bands used to encode a position (the query
        encoder sees ``2 * n_fourier`` features).
    activation : str or callable
        Feed-forward / query-MLP nonlinearity (default GELU). May be a name
        from :data:`~boss_emulator.emulator.ACTIVATIONS`.
    dropout : float
        Dropout rate inside attention and the feed-forward networks. 0.0
        disables it. Active only in training mode; call ``model.eval()`` for
        deterministic inference.

    Notes
    -----
    Activations scale as ``batch * n_queried_positions * d_model``. During
    training only a subsample of positions is queried per step (see
    ``training.py``); full-spectrum reconstruction queries positions in chunks
    to bound memory.
    """

    def __init__(
        self,
        n_features: int,
        n_pixels: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_blocks: int = 4,
        mlp_dim: int = 512,
        n_fourier: int = 16,
        activation: str | Activation = nnx.gelu,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.activation = resolve_activation(activation)
        # Label memory (keys/values): scalar -> token + learned label identity.
        self.value_proj = nnx.Linear(1, d_model, rngs=rngs)
        self.label_embed = nnx.Embed(n_features, d_model, rngs=rngs)
        # Query encoder: Fourier features of the position -> d_model.
        self.query_fc1 = nnx.Linear(2 * n_fourier, d_model, rngs=rngs)
        self.query_fc2 = nnx.Linear(d_model, d_model, rngs=rngs)
        self.blocks = nnx.data([
            _CrossAttnBlock(d_model, n_heads, mlp_dim, self.activation, dropout,
                            rngs=rngs)
            for _ in range(n_blocks)
        ])
        self.ln_out = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, 1, rngs=rngs)
        self.n_features = n_features
        self.n_pixels = n_pixels
        self.d_model = d_model
        self.n_fourier = n_fourier

    def __call__(self, x: jax.Array, coords: jax.Array) -> jax.Array:
        """Predict flux for a batch of labels at a set of positions.

        ``x`` is ``(B, n_features)`` standardised labels; ``coords`` is ``(K,)``
        (or ``(K, 1)``) normalised positions in ``[-1, 1]``. Returns ``(B, K)``
        standardised flux -- pixels are mutually independent, so any subset of
        positions may be queried.
        """
        b = x.shape[0]
        # Label memory (keys/values): value token + learned label identity.
        values = self.value_proj(x[..., None])                # (b, n_feat, d)
        labels = self.label_embed(jnp.arange(self.n_features))  # (n_feat, d)
        kv = values + labels[None]                            # (b, n_feat, d)
        # Position queries from Fourier features, shared across the batch.
        ff = fourier_features(coords, self.n_fourier)          # (K, 2F)
        q = self.query_fc2(self.activation(self.query_fc1(ff)))  # (K, d)
        q = jnp.broadcast_to(q, (b, q.shape[0], self.d_model))  # (b, K, d)
        for block in self.blocks:
            q = block(q, kv)
        out = self.head(self.ln_out(q))                       # (b, K, 1)
        return out[..., 0]                                    # (b, K)
