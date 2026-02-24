#!/usr/bin/env python
"""CLI entry point for mlx-audio-separator."""

import argparse
import json
import logging
import os
import sys


def main():
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(
        description="MLX-native audio stem separation for Apple Silicon.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60),
    )

    parser.add_argument("audio_files", nargs="*", help="The audio file paths or directory to separate.", default=argparse.SUPPRESS)

    try:
        from mlx_audio_separator import __version__
        package_version = __version__
    except Exception:
        package_version = "dev"

    info_params = parser.add_argument_group("Info and Debugging")
    info_params.add_argument("-v", "--version", action="version", version=f"%(prog)s {package_version}")
    info_params.add_argument("-d", "--debug", action="store_true", help="Enable debug logging.")
    info_params.add_argument("-e", "--env_info", action="store_true", help="Print environment information and exit.")
    info_params.add_argument("-l", "--list_models", action="store_true", help="List all supported models and exit.")
    info_params.add_argument("--list_filter", help="Filter/sort: 'name', 'filename', or a stem name (e.g. vocals, drums).")
    info_params.add_argument("--list_limit", type=int, help="Show only top N results.")
    info_params.add_argument("--list_format", choices=["pretty", "json"], default="pretty", help="Output format (default: %(default)s).")
    info_params.add_argument("--log_level", default="info", help="Log level (default: %(default)s).")
    info_params.add_argument("--benchmark", metavar="AUDIO_FILE", help="Benchmark all models against AUDIO_FILE.")
    info_params.add_argument("--benchmark_cooldown", type=float, default=15.0, help="Cooldown between models (default: %(default)s).")
    info_params.add_argument("--benchmark_skip_download", action="store_true", help="Only benchmark already-downloaded models.")
    info_params.add_argument("--benchmark_resume", action="store_true", help="Resume from previous results, skipping completed models.")
    info_params.add_argument("--benchmark_wait_nominal", action="store_true", help="Wait for nominal thermal state between models.")
    info_params.add_argument("--benchmark_repeats", type=int, default=3, help="Timed repeats per model (default: %(default)s).")
    info_params.add_argument("--benchmark_warmup", type=int, default=1, help="Warmup runs per model (default: %(default)s).")
    info_params.add_argument("--benchmark_profile", action="store_true", help="Include per-phase performance profiles in benchmark output.")

    io_params = parser.add_argument_group("Separation I/O Params")
    io_params.add_argument(
        "-m", "--model_filename",
        default="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        help="Model to use for separation (default: %(default)s).",
    )
    io_params.add_argument("--output_format", default="FLAC", help="Output format (default: %(default)s).")
    io_params.add_argument("--output_bitrate", default=None, help="Output bitrate (default: %(default)s).")
    io_params.add_argument("--output_dir", default=None, help="Output directory (default: current dir).")
    io_params.add_argument("--model_file_dir", default="/tmp/audio-separator-models/", help="Model files directory (default: %(default)s).")
    io_params.add_argument("--download_model_only", action="store_true", help="Download model without performing separation.")

    common_params = parser.add_argument_group("Common Separation Parameters")
    common_params.add_argument("--invert_spect", action="store_true", help="Invert secondary stem using spectrogram.")
    common_params.add_argument(
        "--normalization", type=float, default=0.9,
        help="Max peak amplitude for normalization (default: %(default)s).",
    )
    common_params.add_argument(
        "--amplification", type=float, default=0.0,
        help="Min peak amplitude for amplification (default: %(default)s).",
    )
    common_params.add_argument("--single_stem", default=None, help="Output only a single stem.")
    common_params.add_argument("--sample_rate", type=int, default=44100, help="Sample rate (default: %(default)s).")
    common_params.add_argument("--chunk_duration", type=float, default=None, help="Split audio into chunks of this duration in seconds.")
    common_params.add_argument("--custom_output_names", type=json.loads, default=None, help='Custom output names in JSON format.')
    common_params.add_argument(
        "--speed_mode",
        choices=["default", "latency_safe", "latency_safe_v2"],
        default="default",
        help="Performance speed profile (default: %(default)s).",
    )
    common_params.add_argument("--auto_tune_batch", action="store_true", help="Auto-tune batch size for the current model/audio.")
    common_params.add_argument("--tune_probe_seconds", type=float, default=8.0, help="Probe duration for auto-tuner (default: %(default)s).")
    common_params.add_argument("--cache_clear_policy", choices=["aggressive", "deferred"], default="aggressive", help="Cache clear policy (default: %(default)s).")
    common_params.add_argument("--write_workers", type=int, default=1, help="Concurrent stem writer workers (default: %(default)s).")
    common_params.add_argument("--perf_trace", action="store_true", help="Write per-file performance trace metrics.")
    common_params.add_argument("--perf_trace_path", default=None, help="JSONL path for performance trace output.")

    demucs_params = parser.add_argument_group("Demucs Architecture Parameters")
    demucs_params.add_argument("--demucs_segment_size", type=str, default="Default", help="Segment size (default: %(default)s).")
    demucs_params.add_argument("--demucs_shifts", type=int, default=2, help="Number of random shifts (default: %(default)s).")
    demucs_params.add_argument("--demucs_overlap", type=float, default=0.25, help="Overlap ratio (default: %(default)s).")
    demucs_params.add_argument("--demucs_batch_size", type=int, default=8, help="Batch size (default: %(default)s).")
    demucs_params.add_argument(
        "--demucs_segments_enabled", type=bool, default=True,
        help="Enable segment-wise processing (default: %(default)s).",
    )

    mdx_params = parser.add_argument_group("MDX Architecture Parameters")
    mdx_params.add_argument("--mdx_segment_size", type=int, default=256, help="Segment size (default: %(default)s).")
    mdx_params.add_argument("--mdx_overlap", type=float, default=0.25, help="Overlap ratio 0-0.999 (default: %(default)s).")
    mdx_params.add_argument("--mdx_batch_size", type=int, default=1, help="Batch size (default: %(default)s).")
    mdx_params.add_argument("--mdx_hop_length", type=int, default=1024, help="Hop length / stride (default: %(default)s).")
    mdx_params.add_argument("--mdx_enable_denoise", action="store_true", help="Run model twice to reduce noise.")

    mdxc_params = parser.add_argument_group("MDXC Architecture Parameters")
    mdxc_params.add_argument("--mdxc_segment_size", type=int, default=256, help="Segment size (default: %(default)s).")
    mdxc_params.add_argument("--mdxc_override_model_segment_size", action="store_true", help="Override model default segment size.")
    mdxc_params.add_argument("--mdxc_overlap", type=int, default=8, help="Overlap between windows (default: %(default)s).")
    mdxc_params.add_argument("--mdxc_batch_size", type=int, default=1, help="Batch size (default: %(default)s).")
    mdxc_params.add_argument("--mdxc_pitch_shift", type=int, default=0, help="Pitch shift in semitones (default: %(default)s).")

    vr_params = parser.add_argument_group("VR Architecture Parameters")
    vr_params.add_argument("--vr_batch_size", type=int, default=1, help="Batch size (default: %(default)s).")
    vr_params.add_argument("--vr_window_size", type=int, default=512, help="Window size: 320, 512, or 1024 (default: %(default)s).")
    vr_params.add_argument("--vr_aggression", type=int, default=5, help="Extraction intensity -100 to 100 (default: %(default)s).")
    vr_params.add_argument("--vr_enable_tta", action="store_true", help="Enable Test-Time Augmentation.")
    vr_params.add_argument("--vr_enable_post_process", action="store_true", help="Enable artifact post-processing.")
    vr_params.add_argument("--vr_post_process_threshold", type=float, default=0.2, help="Post-process threshold (default: %(default)s).")
    vr_params.add_argument("--vr_high_end_process", action="store_true", help="Mirror missing high frequencies.")

    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    # Set MLX environment variables for performance
    os.environ.setdefault("MLX_USE_FAST_SDP", "1")

    from mlx_audio_separator.core import Separator

    if args.env_info:
        separator = Separator()
        sys.exit(0)

    if args.list_models:
        separator = Separator(info_only=True)

        if args.list_format == "json":
            model_list = separator.list_supported_model_files()
            print(json.dumps(model_list, indent=2))
        else:
            model_list = separator.get_simplified_model_list(filter_sort_by=args.list_filter)

            if args.list_limit:
                model_list = dict(list(model_list.items())[:args.list_limit])

            if model_list:
                # Calculate dynamic column widths
                fn_width = max(len(fn) for fn in model_list)
                type_width = max(len(v["Type"]) for v in model_list.values())
                stems_width = max(len(", ".join(v["Stems"])) for v in model_list.values())

                header = f"  {'Model Filename':<{fn_width}}  {'Arch':<{type_width}}  {'Output Stems (SDR)':<{stems_width}}  Friendly Name"
                print(header)
                print("  " + "-" * (len(header) - 2))

                for filename, info in model_list.items():
                    stems_str = ", ".join(info["Stems"])
                    print(f"  {filename:<{fn_width}}  {info['Type']:<{type_width}}  {stems_str:<{stems_width}}  {info['Name']}")
            else:
                print("No models found matching the filter criteria.")

        sys.exit(0)

    if args.benchmark:
        from mlx_audio_separator.utils.benchmark import run_benchmark

        run_benchmark(
            audio_file=args.benchmark,
            output_dir=args.output_dir,
            model_file_dir=args.model_file_dir,
            cooldown=args.benchmark_cooldown,
            skip_download=args.benchmark_skip_download,
            wait_nominal=args.benchmark_wait_nominal,
            resume=args.benchmark_resume,
            repeats=args.benchmark_repeats,
            warmup=args.benchmark_warmup,
            profile=args.benchmark_profile,
            list_filter=args.list_filter,
            list_limit=args.list_limit,
            log_level=log_level,
            log_formatter=log_formatter,
        )
        sys.exit(0)

    if args.download_model_only:
        separator = Separator(log_formatter=log_formatter, log_level=log_level, model_file_dir=args.model_file_dir)
        separator.download_model_and_data(args.model_filename)
        logger.info(f"Model {args.model_filename} downloaded successfully.")
        sys.exit(0)

    audio_files = list(getattr(args, "audio_files", []))
    if not audio_files:
        parser.print_help()
        sys.exit(1)

    logger.info(f"MLX Audio Separator version {package_version} beginning with input path(s): {', '.join(audio_files)}")

    separator = Separator(
        log_formatter=log_formatter,
        log_level=log_level,
        model_file_dir=args.model_file_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        output_bitrate=args.output_bitrate,
        normalization_threshold=args.normalization,
        amplification_threshold=args.amplification,
        output_single_stem=args.single_stem,
        invert_using_spec=args.invert_spect,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        demucs_params={
            "segment_size": args.demucs_segment_size,
            "shifts": args.demucs_shifts,
            "overlap": args.demucs_overlap,
            "batch_size": args.demucs_batch_size,
            "segments_enabled": args.demucs_segments_enabled,
        },
        mdx_params={
            "segment_size": args.mdx_segment_size,
            "overlap": args.mdx_overlap,
            "batch_size": args.mdx_batch_size,
            "hop_length": args.mdx_hop_length,
            "enable_denoise": args.mdx_enable_denoise,
        },
        mdxc_params={
            "segment_size": args.mdxc_segment_size,
            "batch_size": args.mdxc_batch_size,
            "overlap": args.mdxc_overlap,
            "override_model_segment_size": args.mdxc_override_model_segment_size,
            "pitch_shift": args.mdxc_pitch_shift,
        },
        vr_params={
            "batch_size": args.vr_batch_size,
            "window_size": args.vr_window_size,
            "aggression": args.vr_aggression,
            "enable_tta": args.vr_enable_tta,
            "enable_post_process": args.vr_enable_post_process,
            "post_process_threshold": args.vr_post_process_threshold,
            "high_end_process": args.vr_high_end_process,
        },
        performance_params={
            "speed_mode": args.speed_mode,
            "auto_tune_batch": args.auto_tune_batch,
            "tune_probe_seconds": args.tune_probe_seconds,
            "cache_clear_policy": args.cache_clear_policy,
            "write_workers": args.write_workers,
            "perf_trace": args.perf_trace,
            "perf_trace_path": args.perf_trace_path,
        },
    )

    separator.load_model(model_filename=args.model_filename)
    output_files = separator.separate(audio_files, custom_output_names=args.custom_output_names)
    logger.info(f"Separation complete! Output file(s): {' '.join(output_files)}")


if __name__ == "__main__":
    main()
