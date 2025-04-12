import numpy as np
import librosa
import scipy.signal as sp


class Custom_GeMAPS:
    """
    Custom GeMAPS feature extraction class.
    This class shows how the features of the GeMAPS feature extraction are calculated.

    Attributes:
        _audio_path (str): Path to the audio file to analyze.
    """

    def __init__(
        self, audio_path: str, sample_rate: int = None, mono: bool = True
    ) -> None:
        """
        Initialize the Custom_GeMAPS class.
        Args:
            audio_path (str): Path to the audio file to analyze.
        """
        self._audio_path: str = audio_path
        self._mono: bool = mono
        self.sr: int = sample_rate
        self.y = None

        if not self._mono:
            raise NotImplementedError(
                "Custom GeMAPS feature extraction only supports mono audio files."
            )
        if self._audio_path is None or "":
            raise ValueError("Audio path cannot be None or empty.")

        self.y, self.sr = librosa.load(self._audio_path, sr=self.sr, mono=self._mono)
