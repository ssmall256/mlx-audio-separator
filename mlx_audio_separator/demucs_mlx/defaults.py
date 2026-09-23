"""Shared Demucs inference defaults.

Kept in one place so the CLI, the public API, the separator wrapper and
``apply_model`` cannot drift apart -- which is exactly what happened with the
batch size.
"""

from __future__ import annotations

from pathlib import Path

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


#: VR-arch segments per forward pass.
#:
#: Measured on UVR-BVE-4B_SN-44100-2, macOS 27 / mlx 0.31.2, via metalq
#: (serialized queue, thermal cooldown between jobs):
#:
#:   batch  45 s clip   195 s clip   peak GPU mem
#:       1     3.488 s     17.946 s        3.24 GB
#:       2     2.975 s     16.364 s        5.00 GB
#:       4     3.223 s     15.073 s        7.52 GB
#:       8     4.162 s     18.402 s       12.56 GB
#:
#: Batch 1 is never the fastest. 2 wins on the short clip and 4 on the long
#: one, but their best-case times are within 0.02 s of each other on the long
#: clip while 4 costs another 2.5 GB -- too much to spend by default on a
#: 16 GB machine. Output across all batch sizes differs by at most 3.052e-05,
#: which is exactly one pcm16 LSB: encoding rounding, not divergence.
DEFAULT_VR_BATCH_SIZE = 2

#: Where downloaded MDX/MDXC/VR/Roformer model files are kept.
#:
#: This used to default to `/tmp/audio-separator-models/`. macOS clears `/tmp`
#: on boot, so every reboot cost a multi-gigabyte re-download, and `/tmp` is
#: world-writable: on a shared machine whichever user creates that directory
#: first owns it, and the downloader trusts any file already sitting at the
#: target path without checking it. Every checkpoint load is `weights_only=True`
#: or protobuf parsing, so a planted file means wrong output rather than code
#: execution -- but a cache in a directory another user controls is still the
#: wrong place for it.
DEFAULT_MODEL_FILE_DIR = str(Path.home() / ".cache" / "mlx-audio-separator" / "models")
LEGACY_MODEL_FILE_DIR = "/tmp/audio-separator-models/"
