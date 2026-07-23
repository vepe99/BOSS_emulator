"""Neural spectral emulators built with Flax NNX.

Two architectures map stellar labels to a normalised spectrum of
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

Both take standardised inputs and return standardised outputs; the
de-standardisation lives with the :class:`~boss_emulator.data.Dataset`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

Activation = Callable[[jax.Array], jax.Array]


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
    activation : callable
        Nonlinearity applied between hidden layers (default GELU).
    """

    def __init__(
        self,
        n_features: int,
        n_pixels: int,
        hidden: Sequence[int] = (512, 512, 512),
        activation: Activation = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.data(_mlp_layers(rngs, n_features, hidden, n_pixels))
        self.activation = activation
        self.n_features = n_features
        self.n_pixels = n_pixels

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
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
    """

    def __init__(
        self,
        n_features: int,
        n_pixels: int,
        latent: int = 256,
        branch_hidden: Sequence[int] = (512, 512),
        trunk_hidden: Sequence[int] = (256, 256, 256),
        activation: Activation = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.branch = nnx.data(_mlp_layers(rngs, n_features, branch_hidden, latent))
        self.trunk = nnx.data(_mlp_layers(rngs, 1, trunk_hidden, latent))
        self.bias = nnx.Param(jnp.zeros((n_pixels,)))
        self.activation = activation
        self.latent = latent
        self.n_features = n_features
        self.n_pixels = n_pixels

        # Fixed, normalised pixel coordinates in [-1, 1], shape (n_pixels, 1).
        coords = jnp.linspace(-1.0, 1.0, n_pixels).reshape(-1, 1)
        self.coords = nnx.Variable(coords)

    def _branch(self, x: jax.Array) -> jax.Array:
        for layer in self.branch[:-1]:
            x = self.activation(layer(x))
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


def build_model(
    name: str,
    n_features: int,
    n_pixels: int,
    seed: int = 0,
    **kwargs,
) -> nnx.Module:
    """Factory: construct a model by name ('payne' or 'deeponet')."""
    rngs = nnx.Rngs(seed)
    name = name.lower()
    if name == "payne":
        return PayneMLP(n_features, n_pixels, rngs=rngs, **kwargs)
    if name == "deeponet":
        return DeepONet(n_features, n_pixels, rngs=rngs, **kwargs)
    raise ValueError(f"unknown model {name!r}; choose 'payne' or 'deeponet'")


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
    if hidden:
        if payload["model"] == "payne":
            kwargs["hidden"] = tuple(hidden)
        else:
            kwargs["branch_hidden"] = tuple(hidden)
    model = build_model(
        payload["model"], payload["n_features"], payload["n_pixels"],
        seed=payload.get("seed", 0), **kwargs)
    graphdef, state = nnx.split(model)
    nnx.replace_by_pure_dict(state, payload["params"])
    model = nnx.merge(graphdef, state)
    return model, payload
