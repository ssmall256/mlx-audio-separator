"""Benchmark runner for all supported models."""

import gc
import importlib.util
import json
import os
import platform
import signal
import sys
import time
from datetime import datetime, timezone
from statistics import median

from mlx_audio_separator.utils.performance import clear_mlx_cache

_CORRUPT_MODEL_ERROR_SNIPPETS = (
    "PytorchStreamReader failed reading zip archive",
    "failed finding central directory",
)


def _print_summary_table(results):
    """Print a summary table of benchmark results."""
    if not results:
        print("\nNo benchmark results to display.")
        return

    ok_results = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]

    if ok_results:
        fn_width = max(len(r["filename"]) for r in ok_results)
        fn_width = max(fn_width, len("Model Filename"))
        arch_width = max(len(r["arch"]) for r in ok_results)
        arch_width = max(arch_width, len("Arch"))

        header = f"  {'Model Filename':<{fn_width}}  {'Arch':<{arch_width}}  {'Load':>7}  {'Separate':>9}  {'Stems':>5}  Status"
        print(f"\n{'=' * len(header)}")
        print("  BENCHMARK RESULTS")
        print(f"{'=' * len(header)}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        # Sort by median separate time ascending
        for r in sorted(ok_results, key=lambda x: x["separate_time"]):
            print(
                f"  {r['filename']:<{fn_width}}  {r['arch']:<{arch_width}}"
                f"  {r['load_time']:>6.1f}s  {r['separate_time']:>8.1f}s  {r['stems']:>5}  {r['status']}"
            )

    if err_results:
        print(f"\n  Failed models ({len(err_results)}):")
        for r in err_results:
            print(f"    {r['filename']}: {r['status']}")

    total = len(results)
    ok = len(ok_results)
    print(f"\n  Total: {total} models, {ok} succeeded, {total - ok} failed")

    if ok_results:
        times = [r["separate_time"] for r in ok_results]
        print(f"  Separate time: min={min(times):.1f}s, max={max(times):.1f}s, "
              f"avg={sum(times) / len(times):.1f}s, total={sum(times):.0f}s")


_THERMAL_STATE_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def _get_thermal_state():
    """Get macOS thermal state via NSProcessInfo.thermalState.

    Returns (int, str) — e.g. (0, "nominal"), or None if unavailable.
    """
    try:
        import ctypes
        import ctypes.util

        objc_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc_lib.objc_getClass.restype = ctypes.c_void_p
        objc_lib.sel_registerName.restype = ctypes.c_void_p
        objc_lib.objc_msgSend.restype = ctypes.c_void_p
        objc_lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        cls = objc_lib.objc_getClass(b"NSProcessInfo")
        process_info = objc_lib.objc_msgSend(cls, objc_lib.sel_registerName(b"processInfo"))

        # thermalState returns NSInteger (c_long), not a pointer
        thermal_send = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc_lib)
        )
        state = thermal_send(process_info, objc_lib.sel_registerName(b"thermalState"))
        return (state, _THERMAL_STATE_NAMES.get(state, f"unknown({state})"))
    except Exception:
        return None


def _wait_for_nominal(poll_interval=5.0):
    """Wait until macOS thermal state returns to nominal. Returns wait time in seconds."""
    t0 = time.time()
    result = _get_thermal_state()
    if result is None:
        print("(thermal state unavailable, skipping wait)")
        return 0.0

    state, name = result
    if state == 0:
        return 0.0

    print(f"thermal state: {name}, waiting for nominal...", end="", flush=True)
    while state != 0:
        time.sleep(poll_interval)
        result = _get_thermal_state()
        if result is None:
            break
        state, name = result
        print(".", end="", flush=True)

    waited = time.time() - t0
    print(f" nominal ({waited:.0f}s)")
    return waited


def _save_results(output_path, data):
    """Atomically write results JSON."""
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, output_path)


def _looks_like_corrupt_model_error(exc: Exception) -> bool:
    message = str(exc)
    return any(snippet in message for snippet in _CORRUPT_MODEL_ERROR_SNIPPETS)


def _cleanup_output_files(output_files):
    for output_file in output_files:
        try:
            os.remove(output_file)
        except OSError:
            pass


def _validate_output_files(output_files):
    missing_files = [path for path in output_files if not os.path.isfile(path)]
    empty_files = [path for path in output_files if os.path.isfile(path) and os.path.getsize(path) <= 0]
    return {
        "stems_nonzero": len(output_files) > 0,
        "outputs_exist": len(missing_files) == 0,
        "outputs_nonempty": len(empty_files) == 0,
        "missing_files": missing_files,
        "empty_files": empty_files,
    }


def _demucs_conversion_dependency_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def _enable_strict_benchmark_diagnostics(separator) -> None:
    setter = getattr(separator, "_set_strict_separation_errors", None)
    if callable(setter):
        setter(True)


def run_benchmark(
    audio_file,
    output_dir=None,
    model_file_dir="/tmp/audio-separator-models/",
    cooldown=15.0,
    wait_nominal=False,
    skip_download=False,
    resume=False,
    repeats=3,
    warmup=1,
    profile=False,
    list_filter=None,
    list_limit=None,
    log_level=None,
    log_formatter=None,
):
    """Run benchmark across all supported models.

    Args:
        audio_file: Path to audio file to use for benchmarking.
        output_dir: Directory for output stems and results JSON.
        model_file_dir: Directory where model files are stored.
        cooldown: Seconds to wait between models.
        wait_nominal: Also wait for thermal state to reach nominal after cooldown.
        skip_download: Only benchmark already-downloaded models.
        resume: Resume from previous results file, skipping completed models.
        repeats: Number of timed repeats per model (median reported).
        warmup: Number of untimed warmup runs before repeats.
        profile: Include per-phase profile metrics when available.
        list_filter: Filter models (passed to get_simplified_model_list).
        list_limit: Limit number of models.
        log_level: Logging level.
        log_formatter: Logging formatter.
    """
    from mlx_audio_separator.core import Separator

    if not os.path.isfile(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    repeats = max(1, int(repeats))
    warmup = max(0, int(warmup))

    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "benchmark_results.json")

    # Get model list
    separator = Separator(info_only=True)
    model_list = separator.get_simplified_model_list(filter_sort_by=list_filter)

    if list_limit:
        model_list = dict(list(model_list.items())[:list_limit])

    # Skip download: filter to models that already exist on disk
    if skip_download:
        before = len(model_list)
        model_list = {
            fn: info for fn, info in model_list.items()
            if os.path.isfile(os.path.join(model_file_dir, fn))
        }
        print(f"Skip-download mode: {len(model_list)}/{before} models found on disk.")

    # Resume: load existing results and skip completed
    completed_filenames = set()
    existing_results = []
    if resume and os.path.isfile(results_path):
        with open(results_path) as f:
            existing_data = json.load(f)
        existing_results = existing_data.get("results", [])
        completed_filenames = {r["filename"] for r in existing_results if r["status"] == "ok"}
        print(f"Resume mode: {len(completed_filenames)} models already completed, skipping.")

    # Build work list
    work_list = [(fn, info) for fn, info in model_list.items() if fn not in completed_filenames]

    # Preflight Demucs conversion dependency; skip Demucs models if missing.
    preflight_skipped = []
    if any(info["Type"] == "Demucs" for _, info in work_list) and not _demucs_conversion_dependency_available():
        demucs_reason = (
            "skipped: missing demucs conversion dependency "
            "(install with: pip install 'demucs-mlx[convert]')"
        )
        non_demucs = []
        for filename, info in work_list:
            if info["Type"] == "Demucs":
                preflight_skipped.append({
                    "filename": filename,
                    "friendly_name": info["Name"],
                    "arch": info["Type"],
                    "load_time": 0.0,
                    "separate_time": 0.0,
                    "separate_runs": [],
                    "stems": 0,
                    "status": demucs_reason,
                })
            else:
                non_demucs.append((filename, info))
        work_list = non_demucs
        print(
            f"Demucs preflight: dependency 'demucs' unavailable; "
            f"skipping {len(preflight_skipped)} Demucs model(s)."
        )

    total = len(work_list) + len(completed_filenames) + len(preflight_skipped)

    if not work_list and not preflight_skipped:
        print("No models to benchmark.")
        if existing_results:
            _print_summary_table(existing_results)
        return

    if wait_nominal:
        result = _get_thermal_state()
        if result is None:
            print("Warning: thermal state unavailable on this system, --benchmark_wait_nominal ignored.")
            wait_nominal = False

    print(f"\nBenchmarking {len(work_list)} models against: {audio_file}")
    cooldown_desc = f"{cooldown}s" + (" + wait for nominal thermal" if wait_nominal else "")
    print(f"Cooldown: {cooldown_desc} | Repeats: {repeats} | Warmup: {warmup} | Output: {results_path}\n")

    # Prepare results data
    results_data = {
        "input_file": os.path.abspath(audio_file),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cooldown": cooldown,
        "system": {
            "machine": platform.machine(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "results": list(existing_results) + preflight_skipped,
    }

    all_results = results_data["results"]

    if not work_list:
        print("No runnable models after preflight checks.")
        _save_results(results_path, results_data)
        print(f"\nResults saved to: {results_path}")
        _print_summary_table(all_results)
        return

    # Signal handler for Ctrl+C — print summary and exit
    def handle_interrupt(signum, frame):
        print("\n\nInterrupted! Saving results and printing summary...")
        _save_results(results_path, results_data)
        _print_summary_table(all_results)
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)

    for i, (filename, info) in enumerate(work_list, start=1):
        offset = len(completed_filenames) + len(preflight_skipped)
        print(f"[{offset + i}/{total}] {filename} ({info['Type']})...", end=" ", flush=True)

        result = {
            "filename": filename,
            "friendly_name": info["Name"],
            "arch": info["Type"],
            "load_time": 0.0,
            "separate_time": 0.0,
            "separate_runs": [],
            "stems": 0,
            "status": "error",
        }

        try:
            sep_kwargs = {
                "model_file_dir": model_file_dir,
                "output_dir": output_dir,
                "output_format": "WAV",
            }
            if log_level is not None:
                sep_kwargs["log_level"] = log_level
            if log_formatter is not None:
                sep_kwargs["log_formatter"] = log_formatter

            sep = Separator(**sep_kwargs)
            _enable_strict_benchmark_diagnostics(sep)
            retry = {"attempted": False, "reason": None, "deleted_model_file": False}

            # Load model
            t0 = time.perf_counter()
            try:
                sep.load_model(model_filename=filename)
            except Exception as e:
                if not _looks_like_corrupt_model_error(e):
                    raise
                retry["attempted"] = True
                retry["reason"] = str(e)
                result["retry"] = dict(retry)
                model_path = os.path.join(model_file_dir, filename)
                if os.path.isfile(model_path):
                    os.remove(model_path)
                    retry["deleted_model_file"] = True
                safetensors_path = os.path.join(
                    model_file_dir,
                    f"{os.path.splitext(filename)[0]}.safetensors",
                )
                if os.path.isfile(safetensors_path):
                    os.remove(safetensors_path)
                sep = Separator(**sep_kwargs)
                _enable_strict_benchmark_diagnostics(sep)
                sep.load_model(model_filename=filename)
            load_time = time.perf_counter() - t0
            if retry["attempted"]:
                result["retry"] = dict(retry)

            # Warmup runs (untimed)
            for _ in range(warmup):
                output_files = []
                try:
                    output_files = sep.separate(audio_file)
                finally:
                    _cleanup_output_files(output_files)

            # Timed runs
            run_times = []
            stems = 0
            perf_samples = []
            last_validation = None
            for repeat_idx in range(repeats):
                output_files = []
                try:
                    t0 = time.perf_counter()
                    output_files = sep.separate(audio_file)
                    run_times.append(time.perf_counter() - t0)
                    stems = len(output_files)
                    last_validation = _validate_output_files(output_files)
                    if not (
                        last_validation["stems_nonzero"]
                        and last_validation["outputs_exist"]
                        and last_validation["outputs_nonempty"]
                    ):
                        result["validation"] = last_validation
                        raise RuntimeError(
                            f"invalid output files on repeat {repeat_idx + 1}/{repeats}: "
                            f"stems={stems}, missing={len(last_validation['missing_files'])}, "
                            f"empty={len(last_validation['empty_files'])}"
                        )
                    last_perf = getattr(sep, "last_perf_metrics", None)
                    if profile and last_perf:
                        perf_samples.append(dict(last_perf))
                finally:
                    _cleanup_output_files(output_files)

            separate_time = median(run_times)

            result["load_time"] = round(load_time, 2)
            result["separate_time"] = round(separate_time, 2)
            result["separate_runs"] = [round(x, 3) for x in run_times]
            result["stems"] = stems
            if last_validation is not None:
                result["validation"] = last_validation
            if perf_samples:
                result["profile_samples"] = perf_samples
            result["status"] = "ok"

            print(f"load={load_time:.1f}s, separate_med={separate_time:.1f}s, stems={stems}")

        except Exception as e:
            result["status"] = f"error: {e}"
            result["diagnostic"] = {
                "exception_type": type(e).__name__,
                "message": str(e),
            }
            print(f"FAILED: {e}")

        all_results.append(result)

        # Save after each model (crash-safe)
        _save_results(results_path, results_data)

        # Cleanup
        gc.collect()
        clear_mlx_cache()

        # Cooldown (skip after last model)
        if i < len(work_list):
            if cooldown > 0:
                time.sleep(cooldown)
            if wait_nominal:
                _wait_for_nominal()

    print(f"\nResults saved to: {results_path}")
    _print_summary_table(all_results)
