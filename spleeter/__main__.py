#!/usr/bin/env python
# coding: utf8

"""
Python oneliner script usage.

USAGE: python -m spleeter {train,evaluate,separate} ...

Notes:
    All critical import involving TF, numpy or Pandas are deported to
    command function scope to avoid heavy import on CLI evaluation,
    leading to large bootstraping time.
"""
import json
from functools import partial
from glob import glob
from itertools import product
from os.path import join
from typing import Dict, List, Optional, Tuple

# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Exit, Typer

from . import SpleeterError
from .audio import Codec
from .options import (
    AudioAdapterOption,
    AudioBitrateOption,
    AudioCodecOption,
    AudioDurationOption,
    AudioInputArgument,
    AudioInputOption,
    AudioOffsetOption,
    AudioOutputOption,
    FilenameFormatOption,
    ModelParametersOption,
    VerboseOption,
    VersionOption,
)
from .utils.logging import configure_logger, logger

# pylint: enable=import-error

spleeter: Typer = Typer(add_completion=False, no_args_is_help=True, short_help="-h")
""" CLI application. """


@spleeter.callback()
def default(
    version: bool = VersionOption,
) -> None:
    pass


@spleeter.command(no_args_is_help=True)
def separate(
    deprecated_files: Optional[str] = AudioInputOption,
    files: List[str] = AudioInputArgument,
    adapter: str = AudioAdapterOption,
    bitrate: str = AudioBitrateOption,
    codec: Codec = AudioCodecOption,
    duration: float = AudioDurationOption,
    offset: float = AudioOffsetOption,
    output_path: str = AudioOutputOption,
    filename_format: str = FilenameFormatOption,
    params_filename: str = ModelParametersOption,
    verbose: bool = VerboseOption,
) -> None:
    """
    Separate audio file(s)
    """
    from .audio.adapter import AudioAdapter
    from .separator import Separator

    configure_logger(verbose)
    if deprecated_files is not None:
        logger.error(
            "⚠️ -i option is not supported anymore, audio files must be supplied "
            "using input argument instead (see spleeter separate --help)"
        )
        raise Exit(20)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = Separator(params_filename)

    for filename in files:
        separator.separate_to_file(
            filename,
            output_path,
            audio_adapter=audio_adapter,
            offset=offset,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            filename_format=filename_format,
            synchronous=False,
        )
    separator.join()


EVALUATION_SPLIT: str = "test"
EVALUATION_METRICS_DIRECTORY: str = "metrics"
EVALUATION_INSTRUMENTS: Tuple[str, ...] = ("vocals", "drums", "bass", "other")
EVALUATION_METRICS: Tuple[str, ...] = ("SDR", "SAR", "SIR", "ISR")
EVALUATION_MIXTURE: str = "mixture.wav"
EVALUATION_AUDIO_DIRECTORY: str = "audio"


def _compile_metrics(metrics_output_directory: str) -> Dict:
    """
    Compiles metrics from given directory and returns results as dict.

    Parameters:
        metrics_output_directory (str):
            Directory to get metrics from.

    Returns:
        Dict:
            Compiled metrics as dict.
    """
    import numpy as np
    import pandas as pd  # type: ignore

    songs = glob(join(metrics_output_directory, "test/*.json"))
    index = pd.MultiIndex.from_tuples(
        product(EVALUATION_INSTRUMENTS, EVALUATION_METRICS),
        names=["instrument", "metric"],
    )
    pd.DataFrame([], index=["config1", "config2"], columns=index)
    metrics: Dict = {
        instrument: {k: [] for k in EVALUATION_METRICS}
        for instrument in EVALUATION_INSTRUMENTS
    }
    for song in songs:
        with open(song, "r") as stream:
            data = json.load(stream)
        for target in data["targets"]:
            instrument = target["name"]
            for metric in EVALUATION_METRICS:
                sdr_med = np.median(
                    [
                        frame["metrics"][metric]
                        for frame in target["frames"]
                        if not np.isnan(frame["metrics"][metric])
                    ]
                )
                metrics[instrument][metric].append(sdr_med)
    return metrics


def entrypoint():
    """Application entrypoint."""
    try:
        spleeter()
    except SpleeterError as e:
        logger.error(e)


if __name__ == "__main__":
    entrypoint()
