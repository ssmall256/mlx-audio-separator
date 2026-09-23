"""Run ONE arm in ONE process and print one JSON line.

One process per arm is not fastidiousness. Reassigning a class __call__ or
mixing arms with different tensor shapes in a single process thrashes MLX's
compile cache and inflates every arm together -- silently, because the run
still looks internally consistent. A 0.435 s baseline became 2.19 s that way.

Arms that need a code change are applied as source patches before import, so
nothing leaks between arms.
"""
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import time

SEP = os.environ.get("MLX_AUDIO_SEPARATOR_SRC",
                     str(pathlib.Path(__file__).resolve().parents[2]))
SC = os.path.dirname(os.path.abspath(__file__))
ARM, SECONDS, REPS = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])

# --- env-var arms, set before any import reads them -------------------------
if ARM == "concat_batching":
    os.environ["MLX_AUDIO_SEPARATOR_DEMUCS_APPLY_CONCAT_BATCHING"] = "1"
if ARM in ("unsafe_slice_add", "flush_1_unsafe"):
    os.environ["MLX_AUDIO_SEPARATOR_UNSAFE_SLICE_ADD"] = "1"

sys.path.insert(0, SEP)

# --- source-patch arms ------------------------------------------------------
PATCHES = {
    "flush_1_unsafe": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                "eval_flush_interval = max(8, int(batch_size) * 2)",
                "eval_flush_interval = 1"),
    "flush_1": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                "eval_flush_interval = max(8, int(batch_size) * 2)",
                "eval_flush_interval = 1"),
    "flush_64": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                 "eval_flush_interval = max(8, int(batch_size) * 2)",
                 "eval_flush_interval = 64"),
    "flush_2": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                "eval_flush_interval = max(8, int(batch_size) * 2)",
                "eval_flush_interval = 2"),
    "flush_4": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                "eval_flush_interval = max(8, int(batch_size) * 2)",
                "eval_flush_interval = 4"),
    "flush_16": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                "eval_flush_interval = max(8, int(batch_size) * 2)",
                "eval_flush_interval = 16"),
    "flush_32": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                "eval_flush_interval = max(8, int(batch_size) * 2)",
                "eval_flush_interval = 32"),
    "compiled_flush_1": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                "eval_flush_interval = max(8, int(batch_size) * 2)",
                "eval_flush_interval = 1"),
    "flush_never": ("mlx_audio_separator/demucs_mlx/apply_mlx.py",
                    "eval_flush_interval = max(8, int(batch_size) * 2)",
                    "eval_flush_interval = 10**9"),
    # Ceiling only: keeps every shape identical but skips the data movement,
    # so the numbers are wrong on purpose. It bounds what removing the 16
    # freq-branch DConv layout flips could ever be worth.
    "no_dconv_transpose": ("mlx_audio_separator/demucs_mlx/mlx_hdemucs.py",
                           "y = y.transpose(0, 2, 1, 3).reshape(-1, C, T)",
                           "y = y.reshape(-1, C, T)"),
}


def patched_import(rel_path, old, new, expect):
    """Import a module from patched source, in place of the real one."""
    path = os.path.join(SEP, rel_path)
    src = open(path).read()
    n = src.count(old)
    if n != expect:
        raise SystemExit(f"{ARM}: expected {expect} occurrence(s) of patch anchor, found {n}")
    src = src.replace(old, new)
    name = rel_path[:-3].replace("/", ".")
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


if ARM in PATCHES:
    rel, old, new = PATCHES[ARM]
    expect = 2 if ARM == "no_dconv_transpose" else 1
    if ARM == "no_dconv_transpose":
        # The inverse flip has to go too, or the shapes stop matching.
        path = os.path.join(SEP, rel)
        src = open(path).read()
        assert src.count(old) == 2, src.count(old)
        src = src.replace(old, new)
        inv = "y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)"
        assert src.count(inv) == 2
        src = src.replace(inv, "y = y.reshape(B, C, Fr, T)")
        name = rel[:-3].replace("/", ".")
        spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
        mod = importlib.util.module_from_spec(spec)
        mod.__file__ = path
        sys.modules[name] = mod
        exec(compile(src, path, "exec"), mod.__dict__)
    else:
        patched_import(rel, old, new, expect)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mlx_audio_separator.demucs_mlx.apply_mlx import apply_model  # noqa: E402
from mlx_audio_separator.demucs_mlx.mlx_convert import load_mlx_model  # noqa: E402

model = load_mlx_model("htdemucs", cache_dir=os.environ.get("DEMUCS_CACHE", f"{SC}/cache017"), auto_convert=False)

# The Demucs path runs zero compiled graphs by default -- mx.compile is used
# only in wiener_mlx.py, which htdemucs never reaches (cac=True). Ablation says
# the path is not compute-bound, which is exactly the profile compilation is
# supposed to help: fewer dispatches, fused elementwise chains. Patching the
# class is safe here because this process runs one arm and nothing else.
if ARM in ("compiled_forward", "compiled_flush_1"):
    _sub = model.models[0] if hasattr(model, "models") else model
    _cls = type(_sub)
    _orig_call = _cls.__call__
    _compiled = {}

    def _compiled_call(self, mix):
        key = (id(self), tuple(mix.shape), str(mix.dtype))
        if key not in _compiled:
            _compiled[key] = mx.compile(lambda m, _s=self: _orig_call(_s, m))
        return _compiled[key](mix)

    _cls.__call__ = _compiled_call
audio = np.load(os.environ["AB_CLIP"])[: int(44100 * SECONDS)]
w = mx.array(audio.T)
r = w.mean(axis=0)
x = ((w - r.mean()) / (r.std() + 1e-8))[None]


def once():
    mx.eval(x)
    mx.synchronize()
    t0 = time.perf_counter()
    out = apply_model(model, x, shifts=0, split=True, overlap=0.25, progress=False)
    mx.eval(out)
    mx.synchronize()
    return time.perf_counter() - t0, out


for _ in range(3):
    once()
ts, out = [], None
for _ in range(REPS):
    dt, out = once()
    ts.append(dt)

stems = np.array(out * r.std() + r.mean())[0].astype(np.float32)
np.save(f"{SC}/arm_{ARM}.npy", stems)
print(json.dumps({
    "arm": ARM, "mlx": mx.__version__, "seconds": SECONDS,
    "median": statistics.median(ts), "best": min(ts),
    "spread_pct": statistics.pstdev(ts) / statistics.median(ts) * 100,
    "reps": REPS, "checksum": float(np.abs(stems.astype(np.float64)).sum()),
}))
