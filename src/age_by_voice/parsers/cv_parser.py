import os
from typing import Union, Tuple
import pandas as pd
from uuid import uuid4

from .base_parser import BaseParser, FEATURE_SETS
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

    def _extract_voice_features(self, line: str) -> Tuple[str, str]:
        """ "
        Extract voice features from a given line of the dataset.
        """
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

        age_group = self._convert_text_age_group(text_age)
        gender = self._convert_gender(text_gender)
        clip_id = str(uuid4())
        audio_path = os.path.join(self._audio_path, path)
        voice_name = age_group + "_" + gender + "_" + client_id[:5]
        self._start_parsing_check(voice_name=voice_name)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Audio file {audio_path} does not exist. Please check the path."
            )

        if None in [age_group, gender, clip_id]:
            raise ValueError(f"Invalid values for clip_id: {clip_id}")

        voice = VoiceModel(
            clip_id=clip_id,
            audio_file_name=path,
            voice_name=voice_name,
            voice_age_group=age_group,
            voice_gender=gender,
        )

        # Add the features to the features dataframe
        self._voices = pd.concat(
            [self._voices, pd.DataFrame([voice.model_dump()])],
            ignore_index=True,
        )

        return clip_id, audio_path

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
