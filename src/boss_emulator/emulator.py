"""Neural spectral emulator built with Flax NNX.

:class:`MLP` maps stellar labels to a normalised spectrum of ``n_pixels``
fluxes in one shot (labels -> R^{n_pixels}). It is a plain multilayer
perceptron with optional residual connections and pre-activation LayerNorm.

The model takes standardised inputs and returns standardised outputs; the
de-standardisation lives with the :class:`~boss_emulator.data.Dataset`
(see :func:`predict_spectra` for the ready-made forward recipe).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

Activation = Callable[[jax.Array], jax.Array]

# Named activations, so a hyper-parameter search can pick one by string and it
# survives a checkpoint round-trip (a bare callable is not picklable/portable).
ACTIVATIONS: dict[str, Activation] = {
    "gelu": nnx.gelu,
    "silu": nnx.silu,
    "relu": nnx.relu,
    "tanh": nnx.tanh,
}


def resolve_activation(activation: str | Activation) -> Activation:
    """Map an activation name to its callable (pass callables through)."""
    if callable(activation):
        return activation
    try:
        return ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(
            f"unknown activation {activation!r}; choose from "
            f"{sorted(ACTIVATIONS)}") from None


def _mlp_layers(
    rngs: nnx.Rngs,
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
) -> list[nnx.Linear]:
    """Build a list of dense layers spanning in_dim -> hidden... -> out_dim."""
    dims = [in_dim, *hidden, out_dim]
    return [
        nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
        for i in range(len(dims) - 1)
    ]


class MLP(nnx.Module):
    """Multilayer perceptron: labels -> full spectrum.

    Parameters
    ----------
    n_features : int
        Number of input labels.
    n_pixels : int
        Length of the output spectrum.
    hidden : sequence of int
        Hidden layer widths.
    activation : str or callable
        Nonlinearity applied between hidden layers (default GELU). May be a
        name from :data:`ACTIVATIONS` so it round-trips through a checkpoint.
    dropout : float
        Dropout rate applied to each hidden activation (0.0 disables it,
        and then no Dropout module is created at all). Active only in
        training mode; call ``model.eval()`` for deterministic inference.
    residual : bool
        If True, add a residual (skip) connection around each hidden block
        whose input and output widths match. The first hidden layer projects
        ``n_features -> hidden[0]`` (widths differ, so no skip); every later
        equal-width hidden layer gets ``x = x + block(x)``. Skip connections
        ease optimisation of deeper stacks and reduce depth sensitivity.
    layernorm : bool
        If True, apply pre-activation LayerNorm at each hidden layer
        (``activation(layer(LN(x)))``), which stabilises training of deeper
        or higher-LR configurations.
    """

    def __init__(
        self,
        n_features: int,
        n_pixels: int,
        hidden: Sequence[int] = (512, 512, 512),
        activation: str | Activation = nnx.gelu,
        dropout: float = 0.0,
        residual: bool = False,
        layernorm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.data(_mlp_layers(rngs, n_features, hidden, n_pixels))
        self.activation = resolve_activation(activation)
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None
        self.residual = residual
        # One pre-activation LayerNorm per hidden layer (input widths), or None.
        if layernorm:
            in_dims = [n_features, *hidden]  # input width of each hidden layer
            self.norms = nnx.data(
                [nnx.LayerNorm(d, rngs=rngs) for d in in_dims])
        else:
            self.norms = None
        self.n_features = n_features
        self.n_pixels = n_pixels

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers[:-1]):
            h = self.norms[i](x) if self.norms is not None else x
            h = self.activation(layer(h))
            if self.dropout is not None:
                h = self.dropout(h)
            # Skip connection only where the block preserves the width.
            x = x + h if (self.residual and h.shape[-1] == x.shape[-1]) else h
        return self.layers[-1](x)  # linear output head


# Canonical model name, plus a back-compat alias: checkpoints trained before
# the rename recorded model="payne"; both mean the MLP.
_MLP_NAMES = ("mlp", "payne")


def build_model(
    name: str,
    n_features: int,
    n_pixels: int,
    seed: int = 0,
    **kwargs,
) -> nnx.Module:
    """Factory: construct a model by name.

    Only the MLP is supported (name ``'mlp'``; the legacy alias ``'payne'`` is
    accepted so older checkpoints still load).
    """
    rngs = nnx.Rngs(seed)
    if name.lower() in _MLP_NAMES:
        return MLP(n_features, n_pixels, rngs=rngs, **kwargs)
    raise ValueError(f"unknown model {name!r}; only 'mlp' is supported")


def load_checkpoint(path):
    """Reconstruct a trained model + normalisation stats from a checkpoint.

    Returns ``(model, payload)`` where ``payload`` is the dict written by
    ``training.save_checkpoint`` (holding x/y mean/std, feature_names, ...).
    Use :func:`predict_spectra` to turn raw labels into de-normalised flux.
    """
    import pickle

    with open(path, "rb") as f:
        payload = pickle.load(f)

    kwargs = dict(payload.get("model_kwargs") or {})
    dropout = payload.get("dropout", 0.0)
    if dropout:
        kwargs["dropout"] = dropout
    hidden = payload.get("hidden")
    if hidden:
        kwargs["hidden"] = tuple(hidden)
    model = build_model(
        payload["model"], payload["n_features"], payload["n_pixels"],
        seed=payload.get("seed", 0), **kwargs)
    # Restore only the learnable Param state (what save_checkpoint persists);
    # non-Param state (rng streams) is already correct from build.
    graphdef, param_state, rest = nnx.split(model, nnx.Param, ...)
    nnx.replace_by_pure_dict(param_state, payload["params"])
    model = nnx.merge(graphdef, param_state, rest)
    model.eval()  # deterministic inference (no-op when there is no dropout)
    return model, payload


def predict_spectra(model, payload, X, batch_size: int = 4096) -> np.ndarray:
    """De-normalised spectra for raw labels ``X`` (shape ``(n, n_features)``).

    Applies the checkpoint's input/output standardisation around the model:
    ``flux = model((X - x_mean)/x_std) * y_std + y_mean``. Batched so large
    label sets fit in memory. This is the one place the forward recipe lives;
    ``predict.py`` builds on it.
    """
    x_mean, x_std = payload["x_mean"], payload["x_std"]
    y_mean, y_std = payload["y_mean"], payload["y_std"]
    X = np.asarray(X, dtype="float32")
    out = np.empty((X.shape[0], payload["n_pixels"]), dtype="float32")
    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        xn = (X[start:stop] - x_mean) / x_std
        yn = np.asarray(model(jnp.asarray(xn)))
        out[start:stop] = yn * y_std + y_mean
    return out
