import numpy as np
import librosa
import scipy.signal as sp

GeMAPS_Settings = {
    "AVERAGING_FILTER_LENGTH": 3,
    "WINDOW_SIZE_LONG": 0.060,
    "WINDOW_SIZE_SHORT": 0.020,
    "DEFAULT_HOP_SIZE": 0.010,
    "GAUSSIAN_WINDOW_STD": 0.04,
    "F0_START": 27.5,
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
        n_fft = int(GeMAPS_Settings["WINDOW_SIZE_LONG"] * self.sr)
        hop_length = int(GeMAPS_Settings["DEFAULT_HOP_SIZE"] * self.sr)

        # Compute the Short-Time Fourier Transform (STFT)
        stft_result = librosa.stft(
            self.y, n_fft=n_fft, hop_length=hop_length, window="hann"
        )

        # Compute the magnitude spectrogram
        spectrogram = np.abs(stft_result)

        # Optionally apply logarithmic scaling
        if log_values:
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        return spectrogram
