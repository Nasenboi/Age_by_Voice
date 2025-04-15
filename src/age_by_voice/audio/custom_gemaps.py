import os
from typing import Literal

import librosa
import numpy as np
import pandas as pd
import scipy.signal as sp
import soundfile as sf

from ..models.custom_gemaps_features import *

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

        # cut audio by window and hop size
        self._window_long = int(GeMAPS_Settings["WINDOW_SIZE_LONG"] * self.sr)
        self._window_short = int(GeMAPS_Settings["WINDOW_SIZE_SHORT"] * self.sr)
        self._hop = int(GeMAPS_Settings["DEFAULT_HOP_SIZE"] * self.sr)
        self._time_frames: np.ndarray = librosa.util.frame(
            self.y, frame_length=self._window_long, hop_length=self._hop
        )

        self._stft = librosa.stft(
            self.y,
            n_fft=self._window_long,
            hop_length=self._hop,
            win_length=self._window_long,
        )

        self._f0, self._f0_voiced_flag, self._f0_prob = librosa.pyin(
            self.y,
            sr=self.sr,
            fmin=GeMAPS_Settings["F0_START"],
            fmax=GeMAPS_Settings["F0_END"],
            frame_length=self._window_long,
            hop_length=self._hop,
        )

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
        self, data: np.ndarray, ignore_zero: bool = False, advanced_stats: bool = False
    ) -> dict:
        """
        Smooth a numpy array using a moving average filter and calculate mean and standard deviation.
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

        data_25 = None
        data_80 = None
        data_90 = None
        if advanced_stats:
            data_25 = np.percentile(non_zero_data, 25) if non_zero_data.size > 0 else 0
            data_80 = np.percentile(non_zero_data, 80) if non_zero_data.size > 0 else 0
            data_90 = np.percentile(non_zero_data, 90) if non_zero_data.size > 0 else 0
        return {
            "smoothed_data": smoothed_data,
            "mean": data_mean,
            "std": data_std,
            "25": data_25,
            "80": data_80,
            "90": data_90,
        }

    def _harmonics_energy(self) -> np.ndarray:
        """
        Calculate the energy at harmonics up to the third harmonic (f3).
        Uses the precomputed STFT and librosa.interp_harmonics.

        Returns:
            np.ndarray: A 3D array where each slice along the new axis corresponds to the energy at a harmonic.
        """
        # Frequenzachse für das STFT berechnen
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self._window_long)

        # Definieren Sie die Harmonischen (bis zur dritten)
        harmonics = [2, 3, 4]  # Grundfrequenz (f0), f1, f2, f3

        # Berechnen Sie die Energie an den Harmonischen
        harmonic_energies = librosa.interp_harmonics(
            x=np.abs(self._stft),  # Amplitude des STFT
            freqs=freqs,
            harmonics=harmonics,
            kind="linear",
            fill_value=0,
            axis=-2,  # Frequenzachse
        )

        return harmonic_energies

    """
    Public methods.
    These methods you can use to extract features from the audio signal.
    """

    # -- Time Domain Features --
    # Loudness
    def loudness(self) -> Loudness:
        """
        Calculate the loudness of the audio signal.
        This function calculates the loudness of the audio signal using the RMS method for every frame

        Returns:
            Loudness: A named tuple containing the loudness mean and standard deviation.
        """
        # Calculate the RMS (Root Mean Square) of the audio signal
        rms = librosa.feature.rms(
            y=self.y, frame_length=self._window_long, hop_length=self._hop
        )

        # Smooth the RMS values
        db_rms = librosa.amplitude_to_db(rms[0], ref=np.max)
        smoothed_rms = self._smooth_and_calculate_stats(
            db_rms, ignore_zero=True, advanced_stats=True
        )

        return Loudness(
            loudness_mean=smoothed_rms["mean"],
            loudness_std=smoothed_rms["std"],
            loudness_25=smoothed_rms["25"],
            loudness_80=smoothed_rms["80"],
            loudness_90=smoothed_rms["90"],
        )

    # Zero Crossing Rate
    def zero_crossing_rate(self) -> Zero_Crossing_Rate:
        """
        Calculate the zero crossing rate of the audio signal.
        This function calculates the zero crossing rate of the audio signal using the RMS method for every frame

        Returns:
            Zero_Crossing_Rate: A named tuple containing the zero crossing rate mean and standard deviation.
        """
        # Calculate the zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=self.y, frame_length=self._window_long, hop_length=self._hop
        )

        # Smooth the ZCR values
        smoothed_zcr = self._smooth_and_calculate_stats(zcr[0], ignore_zero=True)

        return Zero_Crossing_Rate(
            zcr_mean=smoothed_zcr["mean"],
            zcr_std=smoothed_zcr["std"],
        )

    # Peaks per Second
    def peaks_per_second(self) -> Peaks_Per_Second:
        """
        Calculate the peaks per second of the audio signal.
        This function calculates the peaks per second of the audio signal using the RMS method for every frame

        Returns:
            Peaks_Per_Second: A named tuple containing the peaks per second mean and standard deviation.
        """
        # Calculate the peaks per second
        pps = []
        pre_max = 3
        post_max = 3
        pre_avg = 2
        post_avg = 2
        delta = 0.5
        wait = 5
        for i in range(self._time_frames.shape[1]):
            # Get the current frame
            frame = self._time_frames[:, i]
            peaks_count = (
                librosa.util.peak_pick(
                    frame,
                    pre_max=pre_max,
                    post_max=post_max,
                    pre_avg=pre_avg,
                    post_avg=post_avg,
                    delta=delta,
                    wait=wait,
                ).shape[0]
                / GeMAPS_Settings["WINDOW_SIZE_LONG"]
            )
            pps.append(peaks_count)
        smoothed_pps = self._smooth_and_calculate_stats(
            np.array(pps), ignore_zero=True, advanced_stats=True
        )
        return Peaks_Per_Second(
            pps_mean=smoothed_pps["mean"],
            pps_std=smoothed_pps["std"],
            pps_25=smoothed_pps["25"],
            pps_80=smoothed_pps["80"],
            pps_90=smoothed_pps["90"],
        )

    # Voiced Segments per second => retrieved from f0 though!

    # Unvoiced Segments per second => retrieved from f0 though!

    # -- Frequency Domain Features --
    def f0(self) -> F0:
        """
        Calculate the fundamental frequency (F0) of the audio signal.
        This function calculates the F0 of the audio signal using the RMS method for every frame

        Returns:
            F0: A named tuple containing the F0 mean and standard deviation.
        """
        valid_f0 = self._f0[~np.isnan(self._f0)]
        # Smooth the F0 values
        smoothed_f0 = self._smooth_and_calculate_stats(
            valid_f0, ignore_zero=True, advanced_stats=True
        )

        return F0(
            f0_mean=smoothed_f0["mean"],
            f0_std=smoothed_f0["std"],
            f0_25=smoothed_f0["25"],
            f0_80=smoothed_f0["80"],
            f0_90=smoothed_f0["90"],
            f0_min=np.min(valid_f0),
            f0_max=np.max(valid_f0),
        )

    # HNR
    def harmonics_to_noice_ratio(self) -> Harmonics_To_Noise_Ratio:
        """
        Calculate the harmonics to noise ratio of the audio signal.
        Returns:
            Harmonics_To_Noise_Ratio: A named tuple containing the HNR mean and standard deviation.
        """
        autocorr = librosa.autocorrelate(self.y)

        f0_idx = (
            int(self.sr / self._f0[np.argmax(self._f0_voiced_flag)])
            if np.any(self._f0_voiced_flag)
            else 0
        )

        if f0_idx > 0 and f0_idx < len(autocorr):
            hnr = 10 * np.log10(autocorr[f0_idx] / np.mean(autocorr[f0_idx + 1 :]))
        else:
            hnr = 0

        smoothed_hnr = self._smooth_and_calculate_stats(
            np.array([hnr]), ignore_zero=True, advanced_stats=True
        )

        return Harmonics_To_Noise_Ratio(
            hnr_mean=smoothed_hnr["mean"],
            hnr_std=smoothed_hnr["std"],
        )

    # Additional Spectral Features
    def additional_spectral_features(self) -> Additional_Spectral_Features:
        """
        Calculate additional spectral features of the audio signal.
        All of them are only for the voiced segments.
        Returns:
            Additional_Spectral_Features: A named tuple containing the spectral features mean and standard deviation.
        """
        # Get S with only voiced lines:
        S = np.abs(self._stft[:, self._f0_voiced_flag])

        centroids = librosa.feature.spectral_centroid(
            S=S,
            sr=self.sr,
        )
        bandwidths = librosa.feature.spectral_bandwidth(
            S=S,
            sr=self.sr,
        )
        contrasts = librosa.feature.spectral_contrast(
            S=S,
            sr=self.sr,
        )
        flatness = librosa.feature.spectral_flatness(
            S=S,
        )
        flatness = librosa.amplitude_to_db(flatness, ref=1.0)
        rolloffs = librosa.feature.spectral_rolloff(
            S=S,
            sr=self.sr,
        )

        # Smooth the spectral features
        centroids = self._smooth_and_calculate_stats(
            centroids[0], ignore_zero=True, advanced_stats=False
        )
        bandwidths = self._smooth_and_calculate_stats(
            bandwidths[0], ignore_zero=True, advanced_stats=False
        )
        contrasts = self._smooth_and_calculate_stats(
            contrasts[0], ignore_zero=True, advanced_stats=False
        )
        flatness = self._smooth_and_calculate_stats(
            flatness[0], ignore_zero=True, advanced_stats=False
        )
        rolloffs = self._smooth_and_calculate_stats(
            rolloffs[0], ignore_zero=True, advanced_stats=False
        )

        return Additional_Spectral_Features(
            spectral_centroid_mean=centroids["mean"],
            spectral_centroid_std=centroids["std"],
            spectral_bandwidth_mean=bandwidths["mean"],
            spectral_bandwidth_std=bandwidths["std"],
            spectral_flatness_mean=flatness["mean"],
            spectral_flatness_std=flatness["std"],
            spectral_rolloff_mean=rolloffs["mean"],
            spectral_rolloff_std=rolloffs["std"],
        )

    # -- Quefrency Domain Features --
    # MFCC 1-1
    def mfcc_1_4(self) -> MFCC_1_4:
        """
        Calculate the first four MFCC (Mel-frequency cepstral coefficients) bands of the audio signal.
        The MFCCs should be calculated using only the voiced segments of the audio signal.
        Returns:
            MFCC_1_4: A named tuple containing the MFCC mean and standard deviation.
        """        
        S = np.abs(self._stft[:, self._f0_voiced_flag])

        S = librosa.feature.melspectrogram(
            S=S,
            sr=self.sr,
            n_mels=4,
        )

        # Calculate MFCCs
        mfccs = librosa.feature.mfcc(
            S=S,
            sr=self.sr,
            n_mfcc=4,
            win_length=self._window_long,
            hop_length=self._hop,
        )

        # Smooth the MFCC values
        smoothed_mfccs = [
            self._smooth_and_calculate_stats(mfcc, ignore_zero=True) for mfcc in mfccs
        ]

        return MFCC_1_4(
            mfcc_1_mean=smoothed_mfccs[0]["mean"],
            mfcc_1_std=smoothed_mfccs[0]["std"],
            mfcc_2_mean=smoothed_mfccs[1]["mean"],
            mfcc_2_std=smoothed_mfccs[1]["std"],
            mfcc_3_mean=smoothed_mfccs[2]["mean"],
            mfcc_3_std=smoothed_mfccs[2]["std"],
            mfcc_4_mean=smoothed_mfccs[3]["mean"],
            mfcc_4_std=smoothed_mfccs[3]["std"],
        )
