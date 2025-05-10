import os
import glob
import tqdm
import pandas as pd
import opensmile
import librosa
from typing import Union, Literal, Tuple

from ..models.custom_gemaps_features import Custom_GeMAPS_Features
from ..audio.custom_gemaps import Custom_GeMAPS
from ..models.gemaps_features import GeMAPS_Features, parse_gemaps_features
from ..models.voice_model import VoiceModel

FEATURE_SETS = Literal["GeMAPSv02", "Custom_GeMAPSv02", "ComParE2016"]


class BaseParser:
    """
    Base class for parsers. This class should be inherited by all parsers.
    Adapt all of the "virtual" methods to the dataset you are using.

    The main usage of the parser is to get a path to the audio files and a path to the dataset description file.
    The output will be one features csv and one voice csv.
    """

    def __init__(
        self,
        dataset_path: str,
        audio_path: str,
        sr=None,
        mono=None,
        save_dir=None,
        feature_set: FEATURE_SETS = "GeMAPSv02",
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
        self._sr = sr
        self._mono = mono
        self._feature_set = feature_set
        if feature_set == "GeMAPSv02":
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.GeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            self._features: pd.DataFrame = pd.DataFrame(
                columns=Custom_GeMAPS_Features.model_fields
            )
        elif feature_set == "Custom_GeMAPSv02":
            self._features: pd.DataFrame = pd.DataFrame(
                columns=GeMAPS_Features.model_fields
            )
        elif feature_set == "ComParE2016":
            self._smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            self._features: pd.DataFrame = pd.DataFrame(
                columns=self._smile.feature_names + ["clip_id"]
            )

        if save_dir:
            self._load_from_temp_file(save_dir)

        self._start_parsing = False

    def parse(
        self,
        save_dir: str = None,
        save_interval: int = 1000,
        num_saves: int = 5,
        extract_audio_features: bool = True,
        first_line_is_header: bool = True,
    ):
        """
        Parse the dataset into the features and voices dataframes.
        Args:
            save_dir (str): Path to save the temporary files.
            save_interval (int): Number of lines to parse before saving.
            num_saves (int): Number of saves to perform.
        """
        with open(self._dataset_path, "r") as file:
            lines = file.readlines()

        if first_line_is_header:
            # delete 1st line
            lines = lines[1:]

        bar = tqdm.tqdm(
            total=len(lines),
            desc="Parsing",
            unit="lines",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - {postfix}",
        )
        for line in lines:
            bar.update(1)
            bar.set_postfix(Clips=len(self._voices))
            try:
                clip_id, audio_path = None, None
                clip_id, audio_path = self._extract_voice_features(line)

                if extract_audio_features:
                    self._extract_audio_features(clip_id, audio_path)

            except Exception as e:
                if clip_id is not None:
                    # if any error occurs, drop the rows with the clip_id
                    self._voices.drop(
                        self._voices[self._voices["clip_id"] == clip_id].index,
                        inplace=True,
                    )
                    self._features.drop(
                        self._features[self._features["clip_id"] == clip_id].index,
                        inplace=True,
                    )
                continue

            # Save dataframes temporarily if save_dir is provided
            if (
                self._start_parsing
                and save_dir
                and len(self._voices) % save_interval == 0
            ):
                self._save_temp_files(save_dir, num_saves)

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

    def _extract_voice_features(self, line: str) -> Tuple[str, str]:
        """ "
        Extract voice features from a given line of the dataset.
        Args:
            line (str): Line from the dataset.
        """
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _extract_audio_features(self, clip_id: str, audio_path: str):
        """
        Extract audio features from a given audio file.
        Append it to the features dataframe.
        """
        features = self._process_audio(
            clip_id=clip_id,
            audio_path=audio_path,
        )
        if self._feature_set == "ComParE2016":
            features = features.iloc[0].to_dict()
        else:
            features = features.model_dump()

        self._features = pd.concat(
            [self._features, pd.DataFrame([features])],
            ignore_index=True,
        )

    def _process_audio(
        self, audio_path: str, clip_id: str
    ) -> Union[GeMAPS_Features, Custom_GeMAPS_Features, pd.DataFrame]:
        """
        Process audio file and extract features.
        This method helps if the audio file is mp3.
        Also this method can keep the logic of the audio processing across all parsers.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            pd.DataFrame: DataFrame with the extracted features.
        """
        if self._feature_set == "Custom_GeMAPSv02":
            custom_gemaps = Custom_GeMAPS(
                audio_path=audio_path, sample_rate=self._sr, mono=self._mono
            )
            return custom_gemaps.custom_gemaps(clip_id=clip_id)
        y, sr = librosa.load(audio_path, sr=self._sr, mono=self._mono)

        smile_features: pd.DataFrame = self._smile.process_signal(y, sr)
        if self._feature_set == "ComParE2016":
            # add clip_id to the features
            smile_features["clip_id"] = clip_id
            return smile_features
        else:
            return parse_gemaps_features(smile_features, clip_id)

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

    def _start_parsing_check(self, voice_name: str = None, clip_id: str = None):
        """
        Checks when to start parsing.
        Should be called on _extract_voice_features.
        Args:
            voice_name (str): Voice name of the current line.
            clip_id (str): Clip ID of the current line.
        """
        if not self._start_parsing:
            if (clip_id is not None and clip_id in self._voices["clip_id"].values) or (
                voice_name is not None
                and voice_name in self._voices["voice_name"].values
            ):
                raise ValueError(f"Clip ID {clip_id} already exists in the dataset.")

            else:
                self._start_parsing = True

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
