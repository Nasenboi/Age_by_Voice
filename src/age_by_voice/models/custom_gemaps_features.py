from pydantic import BaseModel
from typing import Optional, Union
from opensmile import FeatureSet, FeatureLevel
import pandas as pd
from .features_model import FeaturesModel


# -- Time Domain Features --
# Loudness
class Loudness(BaseModel):
    loudness_mean: float
    loudness_std: float
    loudness_25: float
    loudness_80: float
    loudness_90: float


# Zero Crossing Rate
class Zero_Crossing_Rate(BaseModel):
    zcr_mean: float
    zcr_std: float


# Peaks per Second
class Peaks_Per_Second(BaseModel):
    pps_mean: float
    pps_std: float
    pps_25: float
    pps_80: float
    pps_90: float


# Voiced Segments per second => retrieved from f0 though!

# Unvoiced Segments per second => retrieved from f0 though!


# -- Frequency Domain Features --
# HNR
class Harmonics_To_Noise_Ratio(BaseModel):
    hnr_mean: float
    hnr_std: float


# F0
class F0(BaseModel):
    f0_mean: float
    f0_std: float
    f0_25: float
    f0_80: float
    f0_90: float
    f0_min: float
    f0_max: float


# Additional Spectral Features
class Additional_Spectral_Features(BaseModel):
    spectral_centroid_mean: float
    spectral_centroid_std: float

    spectral_bandwidth_mean: float
    spectral_bandwidth_std: float

    spectral_flatness_mean: float
    spectral_flatness_std: float

    spectral_rolloff_mean: float
    spectral_rolloff_std: float


# -- Quefrency Domain Features --
# MFCC_1-4
class MFCC_1_4(BaseModel):
    mfcc_1_mean: float
    mfcc_1_std: float
    mfcc_2_mean: float
    mfcc_2_std: float
    mfcc_3_mean: float
    mfcc_3_std: float
    mfcc_4_mean: float
    mfcc_4_std: float


class Custom_GeMAPS_Features(
    FeaturesModel,
    Loudness,
    Zero_Crossing_Rate,
    Peaks_Per_Second,
    Harmonics_To_Noise_Ratio,
    F0,
    Additional_Spectral_Features,
    MFCC_1_4,
):
    """
    Custom GeMAPS features model.
    This model is based off the GeMAPS features model implemented in opensmile.
    This is an adapted feature list to fulfill the specific task of age estimation.
    """
