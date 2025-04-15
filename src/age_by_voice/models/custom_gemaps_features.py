from pydantic import BaseModel
from typing import Optional, Union
from opensmile import FeatureSet, FeatureLevel
import pandas as pd
from .features_model import FeaturesModel


class Custom_GeMAPS_Features(FeaturesModel):
    """
    Custom GeMAPS features model.
    This model is based off the GeMAPS features model implemented in opensmile.
    This is an adapted feature list to fulfill the specific task of age estimation.
    """
    # -- Time Domain Features --
    # Loudness

    # Energy

    # Zero Crossing Rate

    # Peaks per Second

    # Voiced Segments per second => retrieved from f0 though!

    # Unvoiced Segments per second => retrieved from f0 though!

    # -- Frequency Domain Features --
    # HNR

    # Spectral Slopes
    
    # F0

    # F1

    # F2

    # F3

    # Jitter

    # Shimmer

    # Flux

    # -- Quefrency Domain Features --
    # MFCC1

    # MFCC2

    # MFCC3

    # MFCC4