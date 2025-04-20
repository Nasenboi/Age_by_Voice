import os
from typing import Union
import pandas as pd
import tqdm
import glob
from uuid import uuid4

from .base_parser import BaseParser, FEATURE_SETS
from ..models.features_model import FeaturesModel
from ..models.gemaps_features import GeMAPS_Features
from ..models.custom_gemaps_features import Custom_GeMAPS_Features
from ..audio.custom_gemaps import Custom_GeMAPS
from ..models.voice_model import VoiceModel


class CVParser(BaseParser):
    """
    Parser for the Common Voice dataset.
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
        Class initializer function.
        Do nothing special here, just call the parent class.
        """
        super().__init__(dataset_path, audio_path, sr, mono, save_dir, feature_set)

    def parse(
        self, save_dir: str = None, save_interval: int = 1000, num_saves: int = 5
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

        # delete 1st line
        lines = lines[1:]

        # TSV Structure:
        # client_id	path	sentence_id	sentence	sentence_domain	up_votes	down_votes	age	gender	accents	variant	locale	segment

        bar = tqdm.tqdm(
            total=len(lines),
            desc="Parsing",
            unit="lines",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - {postfix}",
        )
        start_parsing = False
        for line in lines:
            bar.update(1)
            bar.set_postfix(Clips=len(self._voices))
            try:
                parts = line.strip().split("\t")
                # print(parts)
                if len(parts) <= 12:
                    for i in range(13 - len(parts)):
                        parts.append("")
                (
                    client_id,
                    path,
                    scentence_id,
                    sentence,
                    sentence_domain,
                    up_votes,
                    down_votes,
                    text_age,
                    text_gender,
                    accents,
                    variant,
                    locale,
                    segment,
                ) = parts

                if not start_parsing:
                    # check if the clip_id is in the dataframe
                    if scentence_id in self._voices["clip_id"].values:
                        # print(f"Already parsed: {scentence_id}")
                        continue
                    else:
                        start_parsing = True

                age_group = self._convert_text_age_group(text_age)
                gender = self._convert_gender(text_gender)

                clip_id = str(uuid4())

                audio_path = os.path.join(self._audio_path, path)

                if not os.path.exists(audio_path):
                    # print(f"Audio file not found: {audio_path}")
                    continue

                if None in [age_group, gender, clip_id]:
                    # print(f"Missing data for line: {line}")
                    continue

                # Generate voice and features
                # print(f"clip_id: {clip_id}, voice_name: {client_id}, voice_age_group: {age_group}, voice_gender: {gender}")
                voice = VoiceModel(
                    clip_id=clip_id,
                    audio_file_name=path,
                    voice_name=client_id,
                    voice_age_group=age_group,
                    voice_gender=gender,
                )
                features = self._process_audio(
                    clip_id=clip_id,
                    audio_path=audio_path,
                )
                if self._feature_set == "ComParE2016":
                    features = features.iloc[0].to_dict()
                else:
                    features = features.model_dump()

            except Exception as e:
                print(f"Error parsing line:\n {e}")
                continue

            # Add the features to the features dataframe
            self._voices = pd.concat(
                [self._voices, pd.DataFrame([voice.model_dump()])],
                ignore_index=True,
            )
            self._features = pd.concat(
                [self._features, pd.DataFrame([features])],
                ignore_index=True,
            )

            # Save dataframes temporarily if save_dir is provided
            if save_dir and len(self._voices) % save_interval == 0:
                self._save_temp_files(save_dir, num_saves)

    def _convert_text_age_group(self, age: str) -> Union[str, None]:
        # age will either be emty or a string like "twenties"
        if age == "twenties":
            return "20"
        elif age == "thirties":
            return "30"
        elif age == "forties":
            return "40"
        elif age == "fifties":
            return "50"
        elif age == "sixties":
            return "60"
        elif age == "seventies":
            return "70"
        else:
            return None

    def _convert_gender(self, text_gender: str) -> Union[str, None]:
        if text_gender == "male_masculine":
            return "m"
        elif text_gender == "female_feminine":
            return "f"
        else:
            return None
