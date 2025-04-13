import os
import numpy as np
import librosa
import scipy.signal as sp
import soundfile as sf
import parselmouth


GeMAPS_Settings = {
    "AVERAGING_FILTER_LENGTH": 3,
    "WINDOW_SIZE_LONG": 0.060,
    "WINDOW_SIZE_SHORT": 0.020,
    "DEFAULT_HOP_SIZE": 0.010,
    "GAUSSIAN_WINDOW_STD": 0.04,
    "F0_START": 27.5,
    "F0_MIN": 50,
    "F0_MAX": 1000,
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
        Calculate the fundamental frequency (F0) for each frame using librosa's piptrack.
        Notice that the original GeMAPS uses the SHS algorithm.
        To simplify the code, another algorithm is used here, librosas piptrack uses Parabolic Interpolation.
        This way the F0 is calculated in a more rudamentary and rougher way than in the original GeMAPS.
        Smooth the F0 values and compute the mean and standard deviation for the entire audio segment.
        Returns:
            dict: A dictionary containing the smoothed F0 values, mean, and standard deviation.
        """
        # Define the hop length and frame length
        hop_length = int(GeMAPS_Settings["DEFAULT_HOP_SIZE"] * self.sr)
        win_length = int(GeMAPS_Settings["WINDOW_SIZE_LONG"] * self.sr)
        fmin = GeMAPS_Settings["F0_MIN"]
        fmax = GeMAPS_Settings["F0_MAX"]

        # Use librosa's piptrack to estimate F0
        pitches, magnitudes = librosa.piptrack(
            y=self.y,
            sr=self.sr,
            win_length=win_length,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )

        # Extract F0 values for each frame
        f0_values = []
        for i in range(pitches.shape[1]):
            pitch_frame = pitches[:, i]
            magnitude_frame = magnitudes[:, i]
            max_idx = np.argmax(magnitude_frame)
            f0 = pitch_frame[max_idx] if magnitude_frame[max_idx] > 0 else 0
            voicing_prob = magnitude_frame[max_idx] / (np.mean(magnitude_frame) + 1e-10)
            if voicing_prob < 0.7:
                f0 = 0

            f0_values.append(f0)

        # Smooth and calculate statistics for F0 values
        stats = self._smooth_and_calculate_stats(np.array(f0_values), ignore_zero=True)

        return {
            "smoothed_f0_values": stats["smoothed_data"].tolist(),
            "mean": stats["mean"],
            "std": stats["std"],
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
        Calculate jitter (frequency perturbation) for the audio signal using the parselmouth library.
        The original GeMAPS uses the SHS algorithm again, as above.
        We use the praat library to simplify the code and calculations.
        Returns:
            dict: A dictionary containing the jitter local, jitter local absolute, jitter rap, and jitter ppq5.
        """

        point_process = parselmouth.praat.call(
            self._sound,
            "To PointProcess (periodic, cc)",
            GeMAPS_Settings["F0_MIN"],
            GeMAPS_Settings["F0_MAX"],
        )

        # calculate the jitter over time
        jitter_local = parselmouth.praat.call(
            point_process,
            "Get jitter (local)",
            time_range_start,
            time_range_end,
            period_floor,
            period_ceiling,
            maximum_period_factor,
        )

        localabsoluteJitter = parselmouth.praat.call(
            point_process,
            "Get jitter (local, absolute)",
            time_range_start,
            time_range_end,
            period_floor,
            period_ceiling,
            maximum_period_factor,
        )
        rapJitter = parselmouth.praat.call(
            point_process,
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
