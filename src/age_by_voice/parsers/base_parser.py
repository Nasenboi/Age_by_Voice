import os
import glob
import pandas as pd
import opensmile
import librosa
from ..models.features_model import (
    FeaturesModel,
    parse_features,
    FEATURE_SET,
    FEATURE_LEVEL,
)
from ..models.voice_model import VoiceModel


class BaseParser:
    """
    Base class for parsers. This class should be inherited by all parsers.
    Adapt all of the "virtual" methods to the dataset you are using.

    The main usage of the parser is to get a path to the audio files and a path to the dataset description file.
    The output will be one features csv and one voice csv.
    """

    def __init__(
        self, dataset_path: str, audio_path: str, sr=None, mono=None, save_dir=None
    ):
        """
        Class initialize funcion.
        Args:
            dataset_path (str): Path to the dataset description file.
            audio_path (str): Path to the audio files.
        """
        self._dataset_path = dataset_path
        self._audio_path = audio_path
        self._voices: pd.DataFrame = pd.DataFrame(columns=VoiceModel.model_fields)
        self._features: pd.DataFrame = pd.DataFrame(columns=FeaturesModel.model_fields)
        self._sr = sr
        self._mono = mono
        self._smile = opensmile.Smile(
            feature_set=FEATURE_SET, feature_level=FEATURE_LEVEL
        )

        if save_dir:
            self._load_from_temp_file(save_dir)

    def parse(
        self, save_dir: str = None, save_interval: int = 1000, num_saves: int = 5
    ):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def save_features(self, path: str):
        """
        Save features to a csv file.
        Args:
            path (str): Path to save the features csv file.
        """
        self._features.to_csv(path, index=False)

    def save_voices(self, path: str):
        """
        Save voices to a csv file.
        Args:
            path (str): Path to save the voices csv file.
        """
        self._voices.to_csv(path, index=False)

    def _process_audio(self, audio_path: str, clip_id: str) -> FeaturesModel:
        """
        Process audio file and extract features.
        This method helps if the audio file is mp3.
        Also this method can keep the logic of the audio processing across all parsers.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            pd.DataFrame: DataFrame with the extracted features.
        """
        y, sr = librosa.load(audio_path, sr=self._sr, mono=self._mono)
        smile_features: pd.DataFrame = self._smile.process_signal(y, sr)
        features: FeaturesModel = parse_features(smile_features, clip_id)
        return features

    def _load_from_temp_file(self, save_dir: str):
        """
        Load the voices and features dataframes from the given directory.
        Args:
            save_dir (str): Directory to load the files from.
        """
        voices_files = sorted(
            glob.glob(os.path.join(save_dir, "save_voices_*.csv")),
            key=os.path.getmtime,
        )
        features_files = sorted(
            glob.glob(os.path.join(save_dir, "save_features_*.csv")),
            key=os.path.getmtime,
        )

        if voices_files:
            self._voices = pd.read_csv(voices_files[-1])
        if features_files:
            self._features = pd.read_csv(features_files[-1])

    def _save_temp_files(self, save_dir: str, num_saves: int):
        """
        Save the voices and features dataframes temporarily to the given directory.
        Delete the oldest files if the number of saved files exceeds num_saves.
        Args:
            save_dir (str): Directory to save the files.
            num_saves (int): Maximum number of saved files.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save voices dataframe
        voices_file = os.path.join(save_dir, f"save_voices_{len(self._voices)}.csv")
        self._voices.to_csv(voices_file, index=False)

        # Save features dataframe
        features_file = os.path.join(save_dir, f"save_features_{len(self._voices)}.csv")
        self._features.to_csv(features_file, index=False)

        # Manage saved files to ensure num_saves limit
        for file_type in ["voices", "features"]:
            files = sorted(
                glob.glob(os.path.join(save_dir, f"save_{file_type}_*.csv")),
                key=os.path.getmtime,
            )
            while len(files) > num_saves:
                os.remove(files.pop(0))
