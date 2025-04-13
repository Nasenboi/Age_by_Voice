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

    def __init__(self, dataset_path: str, audio_path: str, sr=None, mono=None):
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

    def parse(self, data):
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
