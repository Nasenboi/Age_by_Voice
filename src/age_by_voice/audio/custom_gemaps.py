import os
import numpy as np
import pandas as pd
import librosa
import scipy.signal as sp
import soundfile as sf

GeMAPS_Settings = {
    "AVERAGING_FILTER_LENGTH": 3,
    "WINDOW_SIZE_LONG": 0.060,
    "WINDOW_SIZE_SHORT": 0.020,
    "DEFAULT_HOP_SIZE": 0.010,
    "GAUSSIAN_WINDOW_STD": 0.04,
    "F0_START": 27,
    "F0_END": 1_000,
    "ALPHA_RATIO_LOW": (50, 1_000),
    "ALPHA_RATIO_HIGH": (1_000, 5_000),
    "HAMMERBERG_INDEX_LOW": (0, 2_000),
    "HAMMERBERG_INDEX_HIGH": (2_000, 5_000),
    "SPECTRAL_SLOPE_LOW": (0, 500),
    "SPECTRAL_SLOPE_HIGH": (500, 5_000),
}


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

        self._scale_input()

    def __del__(self):
        """
        Destructor to clean up the class
        """
        pass

    """
    Private methods.
    You dont really need to know what is happening in there.
    """

    def _scale_input(self) -> None:
        """
        Scale the input audio signal.
        This function scales the input audio signal to the range of [-1, 1] and changes the bit depth to 32-bit float.
        """
        # Scale the input audio signal to the range of [-1, 1]
        self.y = librosa.util.normalize(self.y)
        # Change the bit depth to 32-bit float
        if self.y.dtype != np.float32:
            self.y = self.y.astype(np.float32)

    def _smooth_and_calculate_stats(
        self, data: np.ndarray, ignore_zero: bool = False
    ) -> dict:
        """
        Smooth a numpy array using a moving average filter and calculate its mean and standard deviation.
        Args:
            data (np.ndarray): The input array to process.
            ignore_zero (bool): Whether to ignore zero values when calculating statistics.
        Returns:
            dict: A dictionary containing the smoothed array, mean, and standard deviation.
        """
        # Get the filter length from GeMAPS_Settings
        filter_length = GeMAPS_Settings["AVERAGING_FILTER_LENGTH"]

        # Smooth the data using a moving average filter
        smoothed_data = np.convolve(
            data, np.ones(filter_length) / filter_length, mode="same"
        )

        # Optionally ignore zero values for statistics
        if ignore_zero:
            non_zero_data = smoothed_data[smoothed_data != 0]
        else:
            non_zero_data = smoothed_data

        # Calculate mean and standard deviation
        data_mean = np.mean(non_zero_data) if non_zero_data.size > 0 else 0
        data_std = np.std(non_zero_data) if non_zero_data.size > 0 else 0

        return {
            "smoothed_data": smoothed_data,
            "mean": data_mean,
            "std": data_std,
        }

    """
    Public methods.
    These methods you can use to extract features from the audio signal.
    """
    