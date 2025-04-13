import os
import numpy as np
import pandas as pd
import librosa
import scipy.signal as sp
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import opensmile

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

        # store a temp file to open with parselmouth
        self._temp_file_name = "temp.wav"
        sf.write(
            self._temp_file_name,
            self.y,
            self.sr,
            subtype="PCM_24",
        )
        self._sound: parselmouth.Sound = parselmouth.Sound(self._temp_file_name)
        self._pitch = call(
            self._sound,
            "To Pitch",
            0.0,
            GeMAPS_Settings["F0_START"],
            GeMAPS_Settings["F0_END"],
        )
        self._point_process = call(
            self._sound,
            "To PointProcess (periodic, cc)",
            GeMAPS_Settings["F0_START"],
            GeMAPS_Settings["F0_END"],
        )

    def __del__(self):
        """
        Destructor to clean up the class
        """
        try:
            os.remove(self._temp_file_name)
        except Exception as e:
            print(f"Error deleting temporary file: {e}")

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

    def smile(self) -> pd.DataFrame:
        """
        Extract the voice features using the original OpenSMILE GeMAPS algorithm.
        This is only for comparison purposes.
        Returns:
            dict: A dictionary containing the extracted voice features.
        """
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        processed_signal = smile.process_signal(self.y, self.sr)
        return processed_signal

    def dft(self, ts: float = 0.0, log_values: bool = False) -> np.ndarray:
        """
        Compute the DFT of the audio signal on a given time segment.
        Args:
            ts (float): Time segment to compute the DFT.
            log_values (bool): Whether to return the logarithm of the DFT values.
        Returns:
            np.ndarray: 2D array with frequencies and their corresponding magnitudes.
        """
        # Calculate the start and end indices for the time segment
        start_time = int(ts * self.sr)
        end_time = int((ts + GeMAPS_Settings["WINDOW_SIZE_LONG"]) * self.sr)

        # Ensure the time segment is within bounds
        if start_time < 0 or end_time > len(self.y):
            raise ValueError("Time segment is out of bounds.")

        # Extract the segment of the audio signal
        segment = self.y[start_time:end_time]

        # Apply a window function (e.g., Hann window) to the segment
        window = np.hanning(len(segment))
        windowed_segment = segment * window

        # Compute the Discrete Fourier Transform (DFT)
        dft_result = np.fft.rfft(windowed_segment)

        # Compute the magnitude of the DFT
        dft_magnitude = np.abs(dft_result)

        # Compute the frequencies corresponding to the DFT bins
        frequencies = np.fft.rfftfreq(len(windowed_segment), d=1 / self.sr)

        # Optionally apply logarithmic scaling
        if log_values:
            dft_magnitude = 20 * np.log10(
                dft_magnitude + 1e-10
            )  # Add small value to avoid log(0)

        # Combine frequencies and magnitudes into a 2D array
        return np.column_stack((frequencies, dft_magnitude))

    def spectrogram(self, log_values: bool = False) -> np.ndarray:
        """
        Compute the spectrogram of the audio signal.
        Args:
            log_values (bool): Whether to return the logarithm of the spectrogram values.
        Returns:
            np.ndarray: 2D array representing the spectrogram.
        """
        # Define the window size and hop size
        win_length = int(GeMAPS_Settings["WINDOW_SIZE_LONG"] * self.sr)
        hop_length = int(GeMAPS_Settings["DEFAULT_HOP_SIZE"] * self.sr)

        # Compute the Short-Time Fourier Transform (STFT)
        stft_result = librosa.stft(
            self.y, win_length=win_length, hop_length=hop_length, window="hann"
        )

        # Compute the magnitude spectrogram
        spectrogram = np.abs(stft_result)

        # Optionally apply logarithmic scaling
        if log_values:
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        return spectrogram

    def f0(self) -> dict:
        """
        Calculate the fundamental frequency (F0) for each frame using the pre-calculated pitch.
        Smooth the F0 values and compute the mean and standard deviation for the entire audio segment.
        Returns:
            dict: A dictionary containing the smoothed F0 values, mean, and standard deviation.
        """
        unit = "Hertz"
        meanF0 = call(self._pitch, "Get mean", 0, 0, unit)
        stdevF0 = call(self._pitch, "Get standard deviation", 0, 0, unit)

        return {
            "pitch": self._pitch,
            "f0_mean": meanF0,
            "f0_stddev": stdevF0,
        }

    def jitter(
        self,
        time_range_start: float = 0.0,
        time_range_end: float = 0.0,  # if 0.0, use the whole audio
        period_floor: float = 0.0001,
        period_ceiling: float = 0.02,
        maximum_period_factor: float = 1.3,
    ) -> dict:
        """
        Calculate jitter (frequency perturbation) for the audio signal using the pre-calculated point process.
        Returns:
            dict: A dictionary containing the jitter local, jitter local absolute, and jitter rap.
        """
        jitter_local = call(
            self._point_process,
            "Get jitter (local)",
            time_range_start,
            time_range_end,
            period_floor,
            period_ceiling,
            maximum_period_factor,
        )

        localabsoluteJitter = call(
            self._point_process,
            "Get jitter (local, absolute)",
            time_range_start,
            time_range_end,
            period_floor,
            period_ceiling,
            maximum_period_factor,
        )
        rapJitter = call(
            self._point_process,
            "Get jitter (rap)",
            time_range_start,
            time_range_end,
            period_floor,
            period_ceiling,
            maximum_period_factor,
        )

        return {
            "jitter_local": jitter_local,
            "jitter_local_absolute": localabsoluteJitter,
            "jitter_rap": rapJitter,
        }

    def shimmer(
        self,
        time_range_start: float = 0.0,
        time_range_end: float = 0.0,  # if 0.0, use the whole audio
        period_floor: float = 0.0001,
        period_ceiling: float = 0.02,
        maximum_period_factor: float = 1.3,
        maximum_amplitude_factor: float = 1.6,
    ) -> dict:
        """
        Calculate shimmer (amplitude perturbation) for the audio signal using the parselmouth library.
        The original GeMAPS uses the SHS algorithm again, as above.
        We use the praat library to simplify the code and calculations.
        Returns:
            dict: A dictionary containing shimmer metrics.
        """
        local_shimmer = call(
            [self._sound, self._point_process],
            "Get shimmer (local)",
            time_range_start,
            time_range_end,
            period_floor,
            period_ceiling,
            maximum_period_factor,
            maximum_amplitude_factor,
        )
        local_db_shimmer = call(
            [self._sound, self._point_process],
            "Get shimmer (local_dB)",
            time_range_start,
            time_range_end,
            period_floor,
            period_ceiling,
            maximum_period_factor,
            maximum_amplitude_factor,
        )

        return {
            "shimmer_local": local_shimmer,
            "shimmer_local_dB": local_db_shimmer,
        }
