"""Neural spectral emulators built with Flax NNX.

Three architectures map stellar labels to a normalised spectrum of
``n_pixels`` fluxes:

* :class:`PayneMLP` -- a Payne-style multilayer perceptron that emits the
  whole spectrum vector in one shot (labels -> R^{n_pixels}).

* :class:`DeepONet` -- an operator-learning network. A *branch* net encodes
  the labels into ``latent`` coefficients; a *trunk* net encodes each pixel
  coordinate into ``latent`` basis values. The predicted flux at a pixel is
  the inner product of the two (plus a bias), i.e. the spectrum is expanded
  in a learned, coordinate-dependent basis. This decouples "which star"
  (branch) from "where on the spectrum" (trunk) and can generalise across
  the pixel grid.

* :class:`SpectraFormer` -- a Perceiver-style cross-attention decoder. Each
  scalar label becomes a memory token (the keys/values); each pixel owns a
  learned query embedding. Stacked multi-head cross-attention blocks (with
  residual connections + pre-LayerNorm) let every pixel query the label
  memory, and a linear head reads out the flux. Pixels attend only to the
  labels, never to each other, so the whole spectrum is produced in a single
  forward pass and the pixels stay mutually independent -- a learned,
  attention-weighted generalisation of the DeepONet factorisation.

All take standardised inputs and return standardised outputs; the
de-standardisation lives with the :class:`~boss_emulator.data.Dataset`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
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


class PayneMLP(nnx.Module):
    """Payne-style MLP: labels -> full spectrum.

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


class DeepONet(nnx.Module):
    """DeepONet mapping labels -> spectrum over a fixed pixel grid.

    The trunk consumes normalised pixel coordinates in [-1, 1]; these are
    fixed for the grid and cached at construction. ``forward`` therefore
    only needs the labels.

    Parameters
    ----------
    n_features : int
        Number of input labels (branch input dimension).
    n_pixels : int
        Number of pixels (trunk is evaluated at each).
    latent : int
        Shared latent dimension p of branch/trunk (number of basis funcs).
    branch_hidden, trunk_hidden : sequence of int
        Hidden widths of the two sub-networks.
    activation : callable
        Nonlinearity (default GELU).
    dropout : float
        Dropout rate applied to each *branch* hidden activation (the
        label-dependent encoder). The trunk sees only fixed pixel
        coordinates and is left clean. 0.0 disables it (no Dropout module
        created). Active only in training mode; ``model.eval()`` for
        deterministic inference.
    """

    def __init__(
        self,
        n_features: int,
        n_pixels: int,
        latent: int = 256,
        branch_hidden: Sequence[int] = (512, 512),
        trunk_hidden: Sequence[int] = (256, 256, 256),
        activation: Activation = nnx.gelu,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.branch = nnx.data(_mlp_layers(rngs, n_features, branch_hidden, latent))
        self.trunk = nnx.data(_mlp_layers(rngs, 1, trunk_hidden, latent))
        self.bias = nnx.Param(jnp.zeros((n_pixels,)))
        self.activation = activation
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None
        self.latent = latent
        self.n_features = n_features
        self.n_pixels = n_pixels

        # Fixed, normalised pixel coordinates in [-1, 1], shape (n_pixels, 1).
        coords = jnp.linspace(-1.0, 1.0, n_pixels).reshape(-1, 1)
        self.coords = nnx.Variable(coords)

    def _branch(self, x: jax.Array) -> jax.Array:
        for layer in self.branch[:-1]:
            x = self.activation(layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        return self.branch[-1](x)  # (batch, latent)

    def _trunk(self, c: jax.Array) -> jax.Array:
        for layer in self.trunk[:-1]:
            c = self.activation(layer(c))
        return self.activation(self.trunk[-1](c))  # (n_pixels, latent)

    def __call__(self, x: jax.Array) -> jax.Array:
        b = self._branch(x)                 # (batch, latent)
        t = self._trunk(self.coords.value)  # (n_pixels, latent)
        # flux[b, pix] = sum_k branch[b, k] * trunk[pix, k] + bias[pix]
        out = b @ t.T                        # (batch, n_pixels)
        return out + self.bias.value


class _CrossAttnBlock(nnx.Module):
    """One pre-LayerNorm cross-attention + feed-forward block.

    Queries (one per pixel) attend to a fixed label memory (K/V). Both the
    attention and the FFN are wrapped in residual connections; LayerNorm is
    applied to the *inputs* of each sub-layer (pre-LN), which trains more
    stably than post-LN for deep stacks. The label memory is not updated
    here -- only the queries are -- so pixels never attend to one another.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_dim: int,
        activation: Activation,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.ln_q = nnx.LayerNorm(d_model, rngs=rngs)
        self.ln_kv = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_heads, in_features=d_model,
            dropout_rate=dropout, decode=False, rngs=rngs)
        self.ln_ff = nnx.LayerNorm(d_model, rngs=rngs)
        self.fc1 = nnx.Linear(d_model, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, d_model, rngs=rngs)
        self.activation = activation
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None

    def _drop(self, x: jax.Array) -> jax.Array:
        return self.dropout(x) if self.dropout is not None else x

    def __call__(self, q: jax.Array, kv: jax.Array) -> jax.Array:
        # Cross-attention sub-layer (queries attend to the label memory).
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        q = q + self.attn(qn, kvn, kvn)
        # Position-wise feed-forward sub-layer.
        h = self._drop(self.activation(self.fc1(self.ln_ff(q))))
        q = q + self._drop(self.fc2(h))
        return q


class SpectraFormer(nnx.Module):
    """Cross-attention decoder: labels -> spectrum, one shot.

    The ``n_features`` scalar labels are projected to ``d_model`` and given a
    learned per-label embedding, forming a small memory of key/value tokens.
    Each of the ``n_pixels`` output pixels owns a learned query embedding.
    ``n_blocks`` multi-head cross-attention blocks let the pixel queries read
    from the label memory (residual + pre-LayerNorm throughout), and a linear
    head maps each refined query to its scalar flux.

    Because pixels attend only to the labels (never to each other), the model
    is permutation-equivariant over pixels and emits the full spectrum in a
    single pass -- an attention-weighted generalisation of DeepONet.

    Parameters
    ----------
    n_features : int
        Number of input labels (number of memory tokens).
    n_pixels : int
        Length of the output spectrum (number of query tokens).
    d_model : int
        Token/embedding width shared by queries and memory.
    n_heads : int
        Number of attention heads (must divide ``d_model``).
    n_blocks : int
        Number of stacked cross-attention blocks.
    mlp_dim : int
        Hidden width of each block's feed-forward network.
    activation : callable
        Feed-forward nonlinearity (default GELU).
    dropout : float
        Dropout rate applied inside attention and the feed-forward networks.
        0.0 disables it (no Dropout modules created). Active only in training
        mode; call ``model.eval()`` for deterministic inference.

    Notes
    -----
    Activations scale as ``batch * n_pixels * d_model`` (and ``* mlp_dim`` in
    the FFN), which for the full 4479-pixel grid is the memory bottleneck.
    Use a smaller ``--batch-size`` (e.g. 256) than the MLP/DeepONet default.
    """

    def __init__(
        self,
        n_features: int,
        n_pixels: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_blocks: int = 4,
        mlp_dim: int = 512,
        activation: Activation = nnx.gelu,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.value_proj = nnx.Linear(1, d_model, rngs=rngs)   # scalar -> token
        self.label_embed = nnx.Embed(n_features, d_model, rngs=rngs)
        self.query_embed = nnx.Embed(n_pixels, d_model, rngs=rngs)
        self.blocks = nnx.data([
            _CrossAttnBlock(d_model, n_heads, mlp_dim, activation, dropout,
                            rngs=rngs)
            for _ in range(n_blocks)
        ])
        self.ln_out = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, 1, rngs=rngs)
        self.d_model = d_model
        self.n_features = n_features
        self.n_pixels = n_pixels

    def __call__(self, x: jax.Array) -> jax.Array:
        b = x.shape[0]
        # Label memory (keys/values): value token + learned label identity.
        values = self.value_proj(x[..., None])                # (b, n_feat, d)
        labels = self.label_embed(jnp.arange(self.n_features))  # (n_feat, d)
        kv = values + labels[None]                            # (b, n_feat, d)
        # Pixel queries, shared across the batch then refined per sample.
        q = self.query_embed(jnp.arange(self.n_pixels))        # (n_pix, d)
        q = jnp.broadcast_to(q, (b, self.n_pixels, self.d_model))
        for block in self.blocks:
            q = block(q, kv)
        out = self.head(self.ln_out(q))                       # (b, n_pix, 1)
        return out[..., 0]                                    # (b, n_pix)


def build_model(
    name: str,
    n_features: int,
    n_pixels: int,
    seed: int = 0,
    **kwargs,
) -> nnx.Module:
    """Factory: construct a model by name.

    Known names: ``'payne'``, ``'deeponet'``, ``'transformer'`` and
    ``'singlelambda'`` (the coordinate-query cross-attention emulator, whose
    forward is ``model(x, coords)`` rather than ``model(x)``).
    """
    rngs = nnx.Rngs(seed)
    name = name.lower()
    if name == "payne":
        return PayneMLP(n_features, n_pixels, rngs=rngs, **kwargs)
    if name == "deeponet":
        return DeepONet(n_features, n_pixels, rngs=rngs, **kwargs)
    if name == "transformer":
        return SpectraFormer(n_features, n_pixels, rngs=rngs, **kwargs)
    if name == "singlelambda":
        # Deferred import: the submodule imports _CrossAttnBlock from this
        # module at its top level, so importing it here avoids a cycle.
        from boss_emulator.singlelambda.emulator_singlelambda import (
            SpectraFormerSingleLambda)
        return SpectraFormerSingleLambda(
            n_features, n_pixels, rngs=rngs, **kwargs)
    raise ValueError(
        f"unknown model {name!r}; choose 'payne', 'deeponet', "
        f"'transformer' or 'singlelambda'")


def load_checkpoint(path):
    """Reconstruct a trained model + normalisation stats from a checkpoint.

    Returns ``(model, payload)`` where ``payload`` is the dict written by
    ``training.save_checkpoint`` (holding x/y mean/std, feature_names, ...).
    Predict de-normalised flux for raw labels ``X`` (shape (n, n_features))
    with::

        xn = (X - payload['x_mean']) / payload['x_std']
        yn = model(jnp.asarray(xn))
        flux = yn * payload['y_std'] + payload['y_mean']
    """
    import pickle

    with open(path, "rb") as f:
        payload = pickle.load(f)

    hidden = payload.get("hidden")
    kwargs = dict(payload.get("model_kwargs") or {})
    dropout = payload.get("dropout", 0.0)
    if dropout:
        kwargs["dropout"] = dropout
    if hidden:
        if payload["model"] == "payne":
            kwargs["hidden"] = tuple(hidden)
        else:
            kwargs["branch_hidden"] = tuple(hidden)
    model = build_model(
        payload["model"], payload["n_features"], payload["n_pixels"],
        seed=payload.get("seed", 0), **kwargs)
    # Restore only the learnable Param state (what save_checkpoint persists);
    # non-Param state (rng streams, coords) is already correct from build.
    graphdef, param_state, rest = nnx.split(model, nnx.Param, ...)
    nnx.replace_by_pure_dict(param_state, payload["params"])
    model = nnx.merge(graphdef, param_state, rest)
    model.eval()  # deterministic inference (no-op when there is no dropout)
    return model, payload
