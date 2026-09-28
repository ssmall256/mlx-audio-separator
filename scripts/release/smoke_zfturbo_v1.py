"""Small, real-weight release smoke for the catalog's ZFTurbo vocals model.

Install the built wheel into an isolated --target directory, then run this script
through metalq. Its audio files are generated in a temporary directory.
"""

import argparse
import logging
import sys
import tempfile
import wave
from pathlib import Path


def write_stereo_wave(path, *, sample_rate, seconds):
    import numpy as np

    sample_count = sample_rate * seconds
    t = np.arange(sample_count, dtype=np.float32) / sample_rate
    noise = np.random.default_rng(2026).standard_normal((sample_count, 2)).astype(np.float32)
    left = 0.12 * np.sin(2 * np.pi * 220 * t) + 0.03 * np.sin(2 * np.pi * 790 * t) + 0.07 * noise[:, 0]
    right = 0.10 * np.sin(2 * np.pi * 330 * t) + 0.02 * np.sin(2 * np.pi * 1150 * t) + 0.07 * noise[:, 1]
    samples = (np.stack([left, right], axis=1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-target", type=Path, required=True)
    parser.add_argument("--model-file-dir", type=Path, default=Path.home() / ".cache/mlx-audio-separator/models")
    args = parser.parse_args()

    wheel_target = args.wheel_target.resolve()
    sys.path.insert(0, str(wheel_target))

    import mlx_audio_io as mac
    import numpy as np

    import mlx_audio_separator
    from mlx_audio_separator.core import Separator

    assert Path(mlx_audio_separator.__file__).resolve().is_relative_to(wheel_target), "Package did not load from the wheel"
    model_id = "mel-roformer-zfturbo-vocals-v1-mlx"
    model_dir = args.model_file_dir / model_id
    assert (model_dir / "model.safetensors").is_file(), "Published weights must already be cached"
    assert (model_dir / "config.json").is_file(), "Published config must already be cached"

    with tempfile.TemporaryDirectory(prefix="zfturbo-release-smoke-") as temporary:
        root = Path(temporary)
        input_path = root / "mixture-44k.wav"
        write_stereo_wave(input_path, sample_rate=44100, seconds=13)

        separator = Separator(
            log_level=logging.ERROR,
            model_file_dir=str(args.model_file_dir),
            output_dir=str(root / "all-stems"),
            output_format="WAV",
            normalization_threshold=1.0,
        )
        assert model_id in separator.get_simplified_model_list(filter_sort_by="vocals")
        separator.load_model(model_id)
        outputs = separator.separate(str(input_path))
        assert len(outputs) == 2, outputs
        stems = {"Vocals": None, "Instrumental": None}
        for path in outputs:
            name = next((stem for stem in stems if stem in Path(path).name), None)
            assert name is not None and stems[name] is None, path
            assert mac.info(path).frames == 13 * 44100
            assert mac.info(path).sample_rate == 44100
            stems[name], _ = mac.load(path, sr=44100, dtype="float32")
        mixture, _ = mac.load(str(input_path), sr=44100, dtype="float32")
        error = float(np.max(np.abs(stems["Vocals"] + stems["Instrumental"] - mixture)))
        assert error < 2e-4, error
        assert float(np.max(np.abs(stems["Vocals"]))) > 1e-8

        short_path = root / "mixture-48k.wav"
        write_stereo_wave(short_path, sample_rate=48000, seconds=1)
        single = Separator(
            log_level=logging.ERROR,
            model_file_dir=str(args.model_file_dir),
            output_dir=str(root / "single-stem"),
            output_format="WAV",
            output_single_stem="Instrumental",
            sample_rate=48000,
        )
        single.load_model(model_id)
        short_outputs = single.separate(str(short_path))
        assert len(short_outputs) == 1 and "Instrumental" in Path(short_outputs[0]).name
        assert mac.info(short_outputs[0]).sample_rate == 44100
        assert mac.info(short_outputs[0]).frames == 44100

    print("## ✅ ZFTurbo vocals v1 release smoke")
    print(f"**Package:** installed wheel `{mlx_audio_separator.__version__}`")
    print("| Check | Result |")
    print("|---|---|")
    print("| Catalog and cached weights | Passed |")
    print("| 13 s stereo, overlapping chunks | Vocals and Instrumental, 573300 frames each |")
    print(f"| Mixture reconstruction | Max absolute error `{error:.2e}` |")
    print("| 48 kHz input, single Instrumental | 44.1 kHz output, 44100 frames |")
    print("> ✅ Temporary audio files were removed after verification.")


if __name__ == "__main__":
    main()
