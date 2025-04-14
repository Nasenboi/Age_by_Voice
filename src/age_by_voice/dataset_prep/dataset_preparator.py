import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Literal

from ..models.features_model import FeaturesModel
from ..models.voice_model import VoiceModel


class Dataset_Perparator:
    """
    Class to prepare the CV dataset
    """

    def __init__(self, voices_csv_path: str, features_csv_path: str) -> None:
        """
        Constructor for the CV_Preparator class.
        """
        self.voices = pd.read_csv(voices_csv_path)
        self.features = pd.read_csv(features_csv_path)
        self._random_state = 42

        self._check_length()
        self._check_order()

    """
        Public methods
    """

    def prepare_gender_dataset(self, test_size: float = 0.1) -> list:
        """
        prepares the dataset for gender classification.
        The features (X) dataframe contains the 88 features from the GeMAPS feature The labels (y) dataframe contains a column for each gender (either male [1, 0] or female [0, 1])
        The l
        Args:
            test_size (float): Size of the test set.
        Returns:
            list: List of tuples containing the features and labels in a train/test split.
        """
        # Check the balance of the labels (voices.voice_gender should have an equal amount of m and f)
        self._check_balance(feature="gender")
        # create y labels: [1, 0] if male, [0, 1] if female:
        y = self._get_y(feature="gender")
        # create X features: (just drop the clip_id)
        X = self.features.drop(columns=["clip_id"])
        # return the train/test split
        return train_test_split(
            X, y, test_size=test_size, random_state=self._random_state
        )

    def prepare_age_dataset(self, test_size: float = 0.1) -> list:
        """
        prepares the dataset for gender classification.
        The features (X) dataframe contains the 88 features from the GeMAPS feature
        The labels (y) dataframe contains a column for each age group (from 20 to 70, and a one in each row)
        Args:
            test_size (float): Size of the test set.
        Returns:
            list: List of tuples containing the features and labels in a train/test split.
        """

        raise NotImplementedError("This method is not implemented yet!")

    """
        Private methods
    """

    def _check_length(self) -> None:
        """
        Check if the voices and features dataframes have the same length.
        """
        if len(self.voices) != len(self.features):
            raise ValueError(
                f"Length of voices ({len(self.voices)}) and features ({len(self.features)}) do not match."
            )

    def _check_order(self) -> None:
        """
        Check if the clip_ids in the voices and features dataframes are in the same order.
        """
        if not all(self.voices["clip_id"].values == self.features["clip_id"].values):
            # Sort them by clip_id
            self.voices.sort_values("clip_id", inplace=True)
            self.features.sort_values("clip_id", inplace=True)

    def _check_balance(self, feature: Literal["gender"]) -> None:
        """
        Check the balance of the dataset.
        Drop rows if necessary.
        """
        if feature == "gender":
            gender_counts = self.voices["voice_gender"].value_counts()
            min_count = gender_counts.min()
            balanced_voices = pd.concat(
                [
                    self.voices[self.voices["voice_gender"] == "m"].sample(
                        n=min_count, random_state=self._random_state
                    ),
                    self.voices[self.voices["voice_gender"] == "f"].sample(
                        n=min_count, random_state=self._random_state
                    ),
                ]
            )
            self.voices = balanced_voices.reset_index(drop=True)

        self.features = self.features[
            self.features["clip_id"].isin(self.voices["clip_id"])
        ].reset_index(drop=True)

    def _get_y(self, feature: Literal["gender"]) -> pd.DataFrame:
        """
        Get the proper y labels for the dataset.
        Args:
            feature (str): Feature to get the labels for.
        Returns:
            pd.DataFrame: DataFrame with the labels.
        """
        if feature == "gender":
            return pd.DataFrame(
                {
                    "male": (self.voices["voice_gender"] == "m").astype(int),
                    "female": (self.voices["voice_gender"] == "f").astype(int),
                }
            )
