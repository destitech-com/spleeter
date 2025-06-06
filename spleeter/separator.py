#!/usr/bin/env python
# coding: utf8

"""
Module that provides a class wrapper for source separation.

Examples:

```python
>>> from spleeter.separator import Separator
>>> separator = Separator('spleeter:2stems')
>>> separator.separate(waveform, lambda instrument, data: ...)
>>> separator.separate_to_file(...)
```
"""

import atexit
import os
from multiprocessing import Pool
from os.path import basename, dirname, join, splitext
from typing import Any, Dict, Generator, List, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import numpy as np
import tensorflow as tf  # type: ignore

from . import SpleeterError
from .audio import Codec
from .audio.adapter import AudioAdapter
from .audio.convertor import to_stereo
from .model import EstimatorSpecBuilder, InputProviderFactory, model_fn
from .model.provider import ModelProvider
from .types import AudioDescriptor
from .utils.configuration import load_configuration

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def create_estimator(params: Dict) -> tf.Tensor:
    """
    Initialize tensorflow estimator that will perform separation

    Parameters:
        params (Dict):
            A dictionary of parameters for building the model

    Returns:
        tf.Tensor:
            A tensorflow estimator
    """
    # Load model.
    provider: ModelProvider = ModelProvider.default()
    params["model_dir"] = provider.get(params["model_dir"])
    # Setup config
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config = tf.estimator.RunConfig(session_config=session_config)
    # Setup estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=params["model_dir"], params=params, config=config
    )
    return estimator


class Separator(object):
    """A wrapper class for performing separation."""

    def __init__(
        self,
        params_descriptor: str,
        multiprocess: bool = True,
    ) -> None:
        """
        Default constructor.

        Parameters:
            params_descriptor (str):
                Descriptor for TF params to be used.
            multiprocess (bool):
                (Optional) Enable multi-processing.
        """
        self._params = load_configuration(params_descriptor)
        self._sample_rate = self._params["sample_rate"]
        self._tf_graph = tf.Graph()
        self._prediction_generator: Optional[Generator] = None
        self._input_provider = None
        self._builder = None
        self._features = None
        self._session = None
        if multiprocess:
            self._pool: Optional[Any] = Pool()
            atexit.register(self._pool.close)
        else:
            self._pool = None
        self._tasks: List = []
        self.estimator = None

    def _get_prediction_generator(self, data: dict) -> Generator:
        """
        Lazy loading access method for internal prediction generator
        returned by the predict method of a tensorflow estimator.

        Returns:
            Generator:
                Generator of prediction.
        """
        if not self.estimator:
            self.estimator = create_estimator(self._params)

        def get_dataset():
            return tf.data.Dataset.from_tensors(data)

        return self.estimator.predict(get_dataset, yield_single_examples=False)

    def join(self, timeout: int = 200) -> None:
        """
        Wait for all pending tasks to be finished.

        Parameters:
            timeout (int):
                (Optional) Task waiting timeout.
        """
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)


    def _separate_tensorflow(
        self, waveform: np.ndarray, audio_descriptor: AudioDescriptor
    ) -> Dict:
        """
        Performs source separation over the given waveform with tensorflow
        backend.

        Parameters:
            waveform (np.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (AudioDescriptor):
                Audio descriptor to be used.

        Returns:
            Dict:
                Separated waveforms.
        """
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)
            
        print(f"{audio_descriptor} waveform shape: {waveform.shape}, {(waveform.min(), waveform.max())}")
    
        prediction_generator = self._get_prediction_generator(
            {"waveform": waveform}
        )
        # NOTE: perform separation.
        prediction = next(prediction_generator)
        return prediction

    def separate(
        self, waveform: np.ndarray, audio_descriptor: Optional[str] = ""
    ) -> Dict:
        """
        Performs separation on a waveform.

        Parameters:
            waveform (np.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (Optional[str]):
                (Optional) string describing the waveform (e.g. filename).

        Returns:
            Dict:
                Separated waveforms.
        """
        return self._separate_tensorflow(waveform, audio_descriptor)

    def separate_to_file(
        self,
        audio_descriptor: AudioDescriptor,
        destination: str,
        audio_adapter: Optional[AudioAdapter] = None,
        offset: float = 0,
        duration: float = 600.0,
        codec: Codec = Codec.WAV,
        bitrate: str = "128k",
        filename_format: str = "{filename}/{instrument}.{codec}",
        synchronous: bool = True,
    ) -> None:
        """
        Performs source separation and export result to file using
        given audio adapter.

        Filename format should be a Python formattable string that could
        use following parameters :

        - {instrument}
        - {filename}
        - {foldername}
        - {codec}.

        Parameters:
            audio_descriptor (AudioDescriptor):
                Describe song to separate, used by audio adapter to
                retrieve and load audio data, in case of file based
                audio adapter, such descriptor would be a file path.
            destination (str):
                Target directory to write output to.
            audio_adapter (AudioAdapter):
                (Optional) Audio adapter to use for I/O.
            offset (int):
                (Optional) Offset of loaded song.
            duration (float):
                (Optional) Duration of loaded song (default: 600s).
            codec (Codec):
                (Optional) Export codec.
            bitrate (str):
                (Optional) Export bitrate.
            filename_format (str):
                (Optional) Filename format.
            synchronous (bool):
                (Optional) True is should by synchronous.
        """
        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()
        waveform, _ = audio_adapter.load(
            audio_descriptor,
            offset=offset,
            duration=duration,
            sample_rate=self._sample_rate,
        )
        sources = self.separate(waveform, audio_descriptor)
        self.save_to_file(
            sources,
            audio_descriptor,
            destination,
            filename_format,
            codec,
            audio_adapter,
            bitrate,
            synchronous,
        )
        return sources

    def save_to_file(
        self,
        sources: Dict,
        audio_descriptor: AudioDescriptor,
        destination: str,
        filename_format: str = "{filename}/{instrument}.{codec}",
        codec: Codec = Codec.WAV,
        audio_adapter: Optional[AudioAdapter] = None,
        bitrate: str = "128k",
        synchronous: bool = True,
    ) -> None:
        """
        Export dictionary of sources to files.

        Parameters:
            sources (Dict):
                Dictionary of sources to be exported. The keys are the name
                of the instruments, and the values are `N x 2` numpy arrays
                containing the corresponding intrument waveform, as
                returned by the separate method
            audio_descriptor (AudioDescriptor):
                Describe song to separate, used by audio adapter to
                retrieve and load audio data, in case of file based audio
                adapter, such descriptor would be a file path.
            destination (str):
                Target directory to write output to.
            filename_format (str):
                (Optional) Filename format.
            codec (Codec):
                (Optional) Export codec.
            audio_adapter (Optional[AudioAdapter]):
                (Optional) Audio adapter to use for I/O.
            bitrate (str):
                (Optional) Export bitrate.
            synchronous (bool):
                (Optional) True is should by synchronous.
        """
        if audio_adapter is None:
            audio_adapter = AudioAdapter.default()
        foldername = basename(dirname(audio_descriptor))
        filename = splitext(basename(audio_descriptor))[0]
        generated = []
        for instrument, data in sources.items():
            path = join(
                destination,
                filename_format.format(
                    filename=filename,
                    instrument=instrument,
                    foldername=foldername,
                    codec=codec,
                ),
            )
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            if path in generated:
                raise SpleeterError(
                    (
                        f"Separated source path conflict : {path},"
                        "please check your filename format"
                    )
                )
            generated.append(path)
            if self._pool:
                task = self._pool.apply_async(
                    audio_adapter.save, (path, data, self._sample_rate, codec, bitrate)
                )
                self._tasks.append(task)
            else:
                audio_adapter.save(path, data, self._sample_rate, codec, bitrate)
        if synchronous and self._pool:
            self.join()
