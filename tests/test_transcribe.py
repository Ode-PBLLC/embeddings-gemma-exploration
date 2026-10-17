"""Offline: the speech / wordless decision and the word-timing output, with a fake Whisper model."""
from types import SimpleNamespace as NS

import numpy as np
import soundfile as sf

from gallery.transcribe import MIN_WORDS, transcribe


class FakeModel:
    def __init__(self, words):
        self.words = words

    def transcribe(self, audio, **kw):
        assert isinstance(audio, np.ndarray) and kw["word_timestamps"]
        ws = [NS(word=f" {w}", start=s, end=e, probability=p) for w, s, e, p in self.words]
        return iter([NS(words=ws)]), None


def wav(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(16000, dtype="float32"), 16000)
    return p


def test_speech_keeps_timed_words(tmp_path):
    words = [(f"w{i}", i * 0.5, i * 0.5 + 0.4, 0.9) for i in range(MIN_WORDS + 3)]
    speech, out = transcribe(FakeModel(words), wav(tmp_path))
    assert speech and out[0] == ["w0", 0.0, 0.4] and len(out) == MIN_WORDS + 3


def test_few_or_unsure_words_are_wordless(tmp_path):
    unsure = [(f"w{i}", i, i + 0.4, 0.2) for i in range(40)]     # wind misheard as words
    assert transcribe(FakeModel(unsure), wav(tmp_path)) == (False, [])
    few = [(f"w{i}", i, i + 0.4, 0.95) for i in range(MIN_WORDS - 1)]
    assert transcribe(FakeModel(few), wav(tmp_path)) == (False, [])
