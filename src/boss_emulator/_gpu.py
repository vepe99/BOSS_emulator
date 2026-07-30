"""Claim a free GPU via autocvd before JAX initialises (lab policy).

Call :func:`claim_free_gpu` at the very top of an entry-point module, *before*
importing ``jax``. It is a no-op when the device is already chosen
(``CUDA_VISIBLE_DEVICES`` set), when CPU is forced (``JAX_PLATFORMS=cpu``), or
when ``BOSS_NO_AUTOCVD`` is set -- so modules can be imported for CPU tests
without blocking on a free GPU. This module deliberately does not import jax.
"""

import os


def claim_free_gpu(num_gpus: int = 1) -> None:
    """Select ``num_gpus`` free GPU(s) with autocvd unless overridden by env."""
    if (
        os.environ.get("BOSS_NO_AUTOCVD")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("JAX_PLATFORMS", "").lower() == "cpu"
    ):
        return
    from autocvd import autocvd

    autocvd(num_gpus=num_gpus)
