# 0.1.18 Scoped Catalog Release Review

Date: 2026-09-28. Scope: add ZFTurbo Mel-Band-RoFormer Vocals v1 through the
existing MDXC/RoFormer path. This review makes no catalog-wide performance or
separation-quality claim.

## Local Gate

| Check | Result |
|---|---|
| Full test suite | 341 passed, 6 skipped (`metalq` job `mq-aa1fa5`) |
| Ruff and lockfile | Passed |
| Wheel and sdist | Built at version 0.1.18; both passed `twine check` |
| Wheel contents | New loader and catalog entry present; regular dependencies contain no additional model library |
| Fresh pinned download | 67,402,202-byte weights; SHA-256 `ef4aa052845a868cfaff93611477bd8f54d8081bc32f2742a9b3c738f0821191`; config valid; cached reload passed |
| One-chunk reference comparison | Max absolute output difference `1.05e-9` on an 8-second stereo clip (`metalq` job `mq-47c03f`) |
| Built-wheel real-weight smoke | 13-second stereo clip across three overlapping chunks; both stems 573,300 frames; mixture reconstruction error `3.05e-5` (`metalq` job `mq-bd3009`) |
| Resample and stem selection | 48 kHz input yielded one Instrumental file at 44.1 kHz, 44,100 frames (same smoke job) |

The real-weight smoke uses cached weights and removes its temporary audio after
verification. The full test suite covers the shared MDXC short-audio path.

## Pending Publication Checks

- CI on the pushed release commit, including the macOS Python matrix.
- TestPyPI's published-artifact macOS install and packaged-catalog smoke.
- PyPI's published-artifact smoke before the GitHub tag and release.
