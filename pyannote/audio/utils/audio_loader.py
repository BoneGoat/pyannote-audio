# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this this notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Audio loading utility using torchcodec with soundfile fallback.

torchaudio 2.11+ removed list_audio_backends(), info(), and load() functions.
This module provides a unified API using torchcodec (preferred) with soundfile fallback.
"""

__all__ = ["TORCHCODEC_AVAILABLE", "load_audio", "_get_torchaudio_info_obj", "_TorchaudioInfo"]

import io
from typing import NamedTuple, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

try:
    from torchcodec.decoders import AudioDecoder

    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False


# torchaudio.info() returns an object with num_frames, sample_rate, etc. attributes
# The exact return type varies by torchaudio version (AudioMetaData was added then removed)
class _TorchaudioInfo(NamedTuple):
    """Compatible representation of torchaudio.info() return value"""

    num_frames: int
    sample_rate: int
    num_channels: int = 1
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def _get_torchaudio_info_obj(file: dict, backend: str = None) -> _TorchaudioInfo:
    """Get audio info as a compatible NamedTuple.

    Uses torchcodec if available (preferred), falls back to soundfile.

    Args:
        file: Audio file mapping with "audio" key
        backend: Ignored, kept for API compatibility

    Returns:
        _TorchaudioInfo NamedTuple with audio metadata
    """
    if TORCHCODEC_AVAILABLE:
        try:
            decoder = AudioDecoder(file["audio"])
            # Calculate num_frames from duration
            duration = decoder.metadata.duration_seconds
            sample_rate = decoder.metadata.sample_rate
            num_frames = int(duration * sample_rate)

            return _TorchaudioInfo(
                num_frames=num_frames,
                sample_rate=sample_rate,
                num_channels=decoder.metadata.num_channels,
                bits_per_sample=32,  # torchcodec uses float32
                encoding="PCM_F",
            )
        except Exception:
            # Fall through to soundfile if torchcodec fails
            pass

    # Fallback to soundfile
    if isinstance(file["audio"], io.IOBase):
        info = sf.info(file["audio"])
        file["audio"].seek(0)
    else:
        info = sf.info(file["audio"])

    # Map soundfile subtype to bits_per_sample and encoding
    subtype = info.subtype.lower()
    if "16" in subtype:
        bits_per_sample = 16
        encoding = "PCM_S"
    elif "24" in subtype:
        bits_per_sample = 24
        encoding = "PCM_S"
    elif "32" in subtype:
        if "float" in subtype:
            bits_per_sample = 32
            encoding = "PCM_F"
        else:
            bits_per_sample = 32
            encoding = "PCM_S"
    elif "64" in subtype:
        bits_per_sample = 64
        encoding = "PCM_F"
    else:
        bits_per_sample = 16
        encoding = "PCM_S"

    return _TorchaudioInfo(
        num_frames=info.frames,
        sample_rate=info.samplerate,
        num_channels=info.channels,
        bits_per_sample=bits_per_sample,
        encoding=encoding,
    )


def load_audio_with_torchcodec(
    source: str | io.RawIOBase,
    frame_offset: int = 0,
    num_frames: int = -1,
    sample_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio using torchcodec.

    Args:
        source: Path or file-like object
        frame_offset: Starting frame (in samples)
        num_frames: Number of frames to load (-1 = all)
        sample_rate: Target sample rate (None = native)
        num_channels: Target channels (None = native)

    Returns:
        Tuple of (waveform tensor with shape (channels, time), sample_rate)
    """
    if sample_rate is not None or num_channels is not None:
        decoder = AudioDecoder(source, sample_rate=sample_rate, num_channels=num_channels)
    else:
        decoder = AudioDecoder(source)

    if num_frames == -1:
        # Load all samples
        samples = decoder.get_all_samples()
    else:
        # Calculate start/stop from frame_offset
        start_sec = frame_offset / decoder.metadata.sample_rate
        end_sec = start_sec + num_frames / decoder.metadata.sample_rate
        samples = decoder.get_samples_played_in_range(start_sec, end_sec)

    return samples.data, samples.sample_rate


def load_audio_with_soundfile(
    source: str | io.RawIOBase,
    frame_offset: int = 0,
    num_frames: int = -1,
    sample_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio using soundfile (fallback).

    Note: soundfile returns float64 arrays, which are converted to float32.
    """
    if isinstance(source, io.IOBase):
        data, sr = sf.read(source)
    else:
        data, sr = sf.read(source)

    # Convert to float32 (soundfile returns float64)
    data = data.astype(np.float32)

    # Handle cropping if needed
    if frame_offset != 0 or num_frames != -1:
        start = frame_offset
        end = start + num_frames if num_frames != -1 else len(data)
        data = data[start:end]

    # Convert to torch tensor (channels first)
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()

    return tensor, sr


def load_audio(
    source: str | io.RawIOBase,
    frame_offset: int = 0,
    num_frames: int = -1,
    sample_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio using torchcodec with soundfile fallback.

    Args:
        source: Path or file-like object
        frame_offset: Starting frame (in samples)
        num_frames: Number of frames to load (-1 = all)
        sample_rate: Target sample rate (None = native)
        num_channels: Target channels (None = native)

    Returns:
        Tuple of (waveform tensor with shape (channels, time), sample_rate)
    """
    if TORCHCODEC_AVAILABLE:
        return load_audio_with_torchcodec(
            source, frame_offset, num_frames, sample_rate, num_channels
        )
    else:
        return load_audio_with_soundfile(
            source, frame_offset, num_frames, sample_rate, num_channels
        )
