#from keras_separator_correct import CorrectKerasAudioSeparator as AudioSeparator
from pytorch_separator import PyTorchAudioSeparator as AudioSeparator

from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator

MODEL_TO_INST = {
    "spleeter:2stems": ("vocals", "accompaniment"),
}


configuration = "spleeter:2stems"
adapter = AudioAdapter.default()
separatorO = Separator(configuration, multiprocess=False)


test_file = "8Mile.wav"

# waveform, _ = adapter.load(test_file, sample_rate=44100)

# sources = separatorO.separate(waveform, test_file)

# separatorO.save_to_file(
#         sources,
#         "8Mile.wav",
#         destination="output/original",
#     )


# Initialize separator with 2-stems model
separator = AudioSeparator(
    model_path="converted_models_pytorch/2stems",
    config_path="converted_models_pytorch/2stems/config.json"
)

# Separate audio file
separated = separator.separate(test_file)

# Save separated tracks
separator.save_separated_audio(separated, "output/pytorch", "8Mile")