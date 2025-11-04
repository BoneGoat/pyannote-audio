import os
from functools import lru_cache
from typing import BinaryIO, Dict, Optional, Tuple, Type, Union
import warnings

import torch

from pyannote.torchaudio._extension import lazy_import_sox_ext
from pyannote.torchaudio.torio._extension import lazy_import_ffmpeg_ext

from . import soundfile_backend

from .backend import Backend
from .common import AudioMetaData
from .ffmpeg import FFmpegBackend
from .soundfile import SoundfileBackend
from .sox import SoXBackend


@lru_cache(None)
def get_available_backends() -> Dict[str, Type[Backend]]:
    backend_specs: Dict[str, Type[Backend]] = {}
    if lazy_import_ffmpeg_ext().is_available():
        backend_specs["ffmpeg"] = FFmpegBackend
    if lazy_import_sox_ext().is_available():
        backend_specs["sox"] = SoXBackend
    if soundfile_backend._IS_SOUNDFILE_AVAILABLE:
        backend_specs["soundfile"] = SoundfileBackend
    return backend_specs


def get_backend(backend_name, backends) -> Backend:
    if backend := backends.get(backend_name):
        return backend
    else:
        raise ValueError(
            f"Unsupported backend '{backend_name}' specified; ",
            f"please select one of {list(backends.keys())} instead.",
        )


def get_info_func():
    backends = get_available_backends()

    def dispatcher(
        uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], backend_name: Optional[str]
    ) -> Backend:
        if backend_name is not None:
            return get_backend(backend_name, backends)

        for backend in backends.values():
            if backend.can_decode(uri, format):
                return backend
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")

    def info(
        uri: Union[BinaryIO, str, os.PathLike],
        format: Optional[str] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
    ) -> AudioMetaData:
        """Get signal information of an audio file.

        Note:
            When the input type is file-like object, this function cannot
            get the correct length (``num_samples``) for certain formats,
            such as ``vorbis``.
            In this case, the value of ``num_samples`` is ``0``.

        Args:
            uri (path-like object or file-like object):
                Source of audio data. The following types are accepted:

                * ``path-like``: File path or URL.
                * ``file-like``: Object with ``read(size: int) -> bytes`` method,
                  which returns byte string of at most ``size`` length.

            format (str or None, optional):
                If not ``None``, interpreted as hint that may allow backend to override the detected format.
                (Default: ``None``)

            buffer_size (int, optional):
                Size of buffer to use when processing file-like objects, in bytes. (Default: ``4096``)

            backend (str or None, optional):
                I/O backend to use.
                If ``None``, function selects backend given input and available backends.
                Otherwise, must be one of [``"ffmpeg"``, ``"sox"``, ``"soundfile"``],
                with the corresponding backend available.
                (Default: ``None``)

                .. seealso::
                   :ref:`backend`

        Returns:
            AudioMetaData
        """
        backend = dispatcher(uri, format, backend)
        return backend.info(uri, format, buffer_size)

    return info


def get_load_func():
    backends = get_available_backends()

    def dispatcher(
        uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], backend_name: Optional[str]
    ) -> Backend:
        if backend_name is not None:
            return get_backend(backend_name, backends)

        for backend in backends.values():
            if backend.can_decode(uri, format):
                return backend
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")

    def load(
        uri: Union[BinaryIO, str, os.PathLike],
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Load audio data from source.

        .. warning::
            In 2.9, this function's implementation will be changed to use
            :func:`~torchaudio.load_with_torchcodec` under the hood. Some
            parameters like ``normalize``, ``format``, ``buffer_size``, and
            ``backend`` will be ignored. We recommend that you port your code to
            rely directly on TorchCodec's decoder instead:
            https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.

        By default (``normalize=True``, ``channels_first=True``), this function returns Tensor with
        ``float32`` dtype, and the shape of `[channel, time]`.

        Note:
            The formats this function can handle depend on the availability of backends.
            Please use the following functions to fetch the supported formats.

            - FFmpeg: :py:func:`torchaudio.utils.ffmpeg_utils.get_audio_decoders`
            - Sox: :py:func:`torchaudio.utils.sox_utils.list_read_formats`
            - SoundFile: Refer to `the official document <https://pysoundfile.readthedocs.io/>`__.

        .. warning::

            ``normalize`` argument does not perform volume normalization.
            It only converts the sample type to `torch.float32` from the native sample
            type.

            When the input format is WAV with integer type, such as 32-bit signed integer, 16-bit
            signed integer, 24-bit signed integer, and 8-bit unsigned integer, by providing ``normalize=False``,
            this function can return integer Tensor, where the samples are expressed within the whole range
            of the corresponding dtype, that is, ``int32`` tensor for 32-bit signed PCM,
            ``int16`` for 16-bit signed PCM and ``uint8`` for 8-bit unsigned PCM. Since torch does not
            support ``int24`` dtype, 24-bit signed PCM are converted to ``int32`` tensors.

            ``normalize`` argument has no effect on 32-bit floating-point WAV and other formats, such as
            ``flac`` and ``mp3``.

            For these formats, this function always returns ``float32`` Tensor with values.


        Args:
            uri (path-like object or file-like object):
                Source of audio data.
            frame_offset (int, optional):
                Number of frames to skip before start reading data.
            num_frames (int, optional):
                Maximum number of frames to read. ``-1`` reads all the remaining samples,
                starting from ``frame_offset``.
                This function may return the less number of frames if there is not enough
                frames in the given file.
            normalize (bool, optional):
                When ``True``, this function converts the native sample type to ``float32``.
                Default: ``True``.

                If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
                integer type.
                This argument has no effect for formats other than integer WAV type.

            channels_first (bool, optional):
                When True, the returned Tensor has dimension `[channel, time]`.
                Otherwise, the returned Tensor's dimension is `[time, channel]`.

            format (str or None, optional):
                If not ``None``, interpreted as hint that may allow backend to override the detected format.
                (Default: ``None``)

            buffer_size (int, optional):
                Size of buffer to use when processing file-like objects, in bytes. (Default: ``4096``)

            backend (str or None, optional):
                I/O backend to use.
                If ``None``, function selects backend given input and available backends.
                Otherwise, must be one of [``"ffmpeg"``, ``"sox"``, ``"soundfile"``],
                with the corresponding backend being available. (Default: ``None``)

                .. seealso::
                   :ref:`backend`

        Returns:
            (torch.Tensor, int): Resulting Tensor and sample rate.
                If the input file has integer wav format and normalization is off, then it has
                integer type, else ``float32`` type. If ``channels_first=True``, it has
                `[channel, time]` else `[time, channel]`.
        """
        warnings.warn(
            "In 2.9, this function's implementation will be changed to use "
            "torchaudio.load_with_torchcodec` under the hood. Some "
            "parameters like ``normalize``, ``format``, ``buffer_size``, and "
            "``backend`` will be ignored. We recommend that you port your code to "
            "rely directly on TorchCodec's decoder instead: "
            "https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder."
        )
        backend = dispatcher(uri, format, backend)
        return backend.load(uri, frame_offset, num_frames, normalize, channels_first, format, buffer_size)

    return load