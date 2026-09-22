"""Shared Demucs inference defaults.

Kept in one place so the CLI, the public API, the separator wrapper and
``apply_model`` cannot drift apart -- which is exactly what happened with the
batch size.
"""

from __future__ import annotations

#: Segments processed per forward pass.
#:
#: Measured on a 128 GB machine, htdemucs, macOS 27 / mlx 0.31.2:
#:
#:   batch  45 s clip   195 s clip   peak memory (195 s)
#:       2     0.872 s      3.901 s              4.75 GB
#:       4          --      5.584 s              5.43 GB
#:       8     1.870 s      5.524 s              9.20 GB
#:      12          --     40.119 s             13.31 GB
#:
#: 2 is both the fastest and the smallest; 8 is roughly 2x slower on shorter
#: inputs at nearly double the memory, and 12 collapses. Smaller Macs fare
#: worse still. demucs-mlx reached the same value independently (see its
#: commit 12f5881, "avoid memory thrashing on 16-36 GB Macs").
DEFAULT_BATCH_SIZE = 2

#: Seed for the shift-trick offsets. A fixed default makes repeated runs on the
#: same input reproduce; pass an explicit seed, or ``None``, to vary per run.
DEFAULT_SHIFT_SEED = 0
