import numpy as np

from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator


MODEL_TO_INST = {
    "spleeter:2stems": ("vocals", "accompaniment"),
}


def test_separate(test_file):
    configuration = "spleeter:2stems"
    instruments = MODEL_TO_INST[configuration]
    adapter = AudioAdapter.default()
    waveform, _ = adapter.load(test_file, sample_rate=44100)

    separator = Separator(configuration, multiprocess=False)
    sources = separator.separate(waveform, test_file)
    separator.save_to_file(
            sources,
            test_file,
            destination="test_output",
        )

    assert len(sources) == len(instruments)
    for instrument in instruments:
        assert instrument in sources
    for instrument in instruments:
        track = sources[instrument]
        assert waveform.shape[:-1] == track.shape[:-1]
        assert not np.allclose(waveform, track)
        for compared in instruments:
            if instrument != compared:
                assert not np.allclose(track, sources[compared])


if __name__ == "__main__":
    #test_separate("audio_example.mp3")
    #test_separate("audio_example_mono.mp3")
    test_separate("8Mile.wav")