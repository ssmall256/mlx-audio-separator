"""Audio chunking utilities for processing large audio files to prevent OOM errors."""

import logging
import os
from typing import List

import mlx_audio_io as mac
import numpy as np


class AudioChunker:
    """
    Handles splitting and merging of large audio files using mlx-audio-io.

    This class provides utilities to:
    - Split large audio files into fixed-duration chunks
    - Merge processed chunks back together with simple concatenation
    - Determine if a file should be chunked based on its duration
    """

    def __init__(self, chunk_duration_seconds: float, logger: logging.Logger = None):
        self.chunk_duration_seconds = chunk_duration_seconds
        self.logger = logger or logging.getLogger(__name__)

    def split_audio(self, input_path: str, output_dir: str, sample_rate: int = 44100) -> List[str]:
        """Split audio file into fixed-size chunks.

        Args:
            input_path: Path to the input audio file
            output_dir: Directory where chunk files will be saved
            sample_rate: Target sample rate for loading

        Returns:
            List of paths to the created chunk files
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        os.makedirs(output_dir, exist_ok=True)

        self.logger.debug(f"Loading audio file: {input_path}")
        audio, sr = mac.load(str(input_path), sr=sample_rate, dtype="float32")
        audio_np = np.array(audio, copy=False)

        # audio_np shape: (frames, channels)
        total_frames = audio_np.shape[0]
        chunk_frames = int(self.chunk_duration_seconds * sr)

        num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
        self.logger.info(
            f"Splitting {total_frames / sr:.1f}s audio into {num_chunks} chunks "
            f"of {self.chunk_duration_seconds:.1f}s each"
        )

        _, ext = os.path.splitext(input_path)
        if not ext:
            ext = ".wav"

        chunk_paths = []
        for i in range(num_chunks):
            start = i * chunk_frames
            end = min(start + chunk_frames, total_frames)
            chunk = audio_np[start:end]

            chunk_filename = f"chunk_{i:04d}{ext}"
            chunk_path = os.path.join(output_dir, chunk_filename)

            self.logger.debug(
                f"Exporting chunk {i + 1}/{num_chunks}: "
                f"{start / sr:.1f}s - {end / sr:.1f}s to {chunk_path}"
            )
            mac.save(str(chunk_path), chunk, sr)
            chunk_paths.append(chunk_path)

        return chunk_paths

    def merge_chunks(self, chunk_paths: List[str], output_path: str, sample_rate: int = 44100) -> str:
        """Merge processed chunks with simple concatenation.

        Args:
            chunk_paths: List of paths to chunk files to merge
            output_path: Path where the merged output will be saved
            sample_rate: Sample rate for loading chunks

        Returns:
            Path to the merged output file
        """
        if not chunk_paths:
            raise ValueError("Cannot merge empty list of chunks")

        for chunk_path in chunk_paths:
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        self.logger.info(f"Merging {len(chunk_paths)} chunks into {output_path}")

        chunks = []
        for i, chunk_path in enumerate(chunk_paths):
            self.logger.debug(f"Loading chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
            audio, sr = mac.load(str(chunk_path), sr=sample_rate, dtype="float32")
            chunks.append(np.array(audio, copy=False))

        combined = np.concatenate(chunks, axis=0)
        total_duration = combined.shape[0] / sample_rate

        self.logger.info(f"Exporting merged audio ({total_duration:.1f}s) to {output_path}")
        mac.save(str(output_path), combined, sample_rate)

        return output_path

    def should_chunk(self, audio_duration_seconds: float) -> bool:
        """Determine if file is large enough to benefit from chunking."""
        return audio_duration_seconds > self.chunk_duration_seconds
