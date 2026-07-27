"""Coordinate-query ("single wavelength") cross-attention emulator.

A variant of :class:`boss_emulator.emulator.SpectraFormer` whose queries come
from a *continuous* pixel position (through Fourier features) instead of a
learned per-pixel embedding, so the spectrum is produced one position at a
time: the model maps ``(labels, position) -> flux`` and a full spectrum is
obtained by querying every position. See :mod:`.emulator_singlelambda`.
"""

from boss_emulator.singlelambda.emulator_singlelambda import (
    SpectraFormerSingleLambda,
    fourier_features,
)

__all__ = ["SpectraFormerSingleLambda", "fourier_features"]
