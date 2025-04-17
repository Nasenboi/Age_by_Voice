from pydantic import BaseModel
from typing import Optional, Union
from opensmile import FeatureSet, FeatureLevel
import pandas as pd
from .features_model import FeaturesModel


# -- Time Domain Features --
# Loudness
class Loudness(BaseModel):
    loudness_mean: Optional[float] = None
    loudness_std: Optional[float] = None
    loudness_25: Optional[float] = None
    loudness_80: Optional[float] = None
    loudness_90: Optional[float] = None


# Zero Crossing Rate
class Zero_Crossing_Rate(BaseModel):
    zcr_mean: Optional[float] = None
    zcr_std: Optional[float] = None


# Peaks per Second
class Peaks_Per_Second(BaseModel):
    pps_mean: Optional[float] = None
    pps_std: Optional[float] = None
    pps_25: Optional[float] = None
    pps_80: Optional[float] = None
    pps_90: Optional[float] = None


# Voiced Segments per second => retrieved from f0 though!

# Unvoiced Segments per second => retrieved from f0 though!


# -- Frequency Domain Features --
# HNR
class Harmonics_To_Noise_Ratio(BaseModel):
    hnr_mean: Optional[float] = None


# F0
class F0(BaseModel):
    f0_mean: Optional[float] = None
    f0_std: Optional[float] = None
    f0_25: Optional[float] = None
    f0_80: Optional[float] = None
    f0_90: Optional[float] = None
    f0_min: Optional[float] = None
    f0_max: Optional[float] = None


# Additional Spectral Features
class Additional_Spectral_Features(BaseModel):
    spectral_centroid_mean: Optional[float] = None
    spectral_centroid_std: Optional[float] = None

    spectral_bandwidth_mean: Optional[float] = None
    spectral_bandwidth_std: Optional[float] = None

    spectral_flatness_mean: Optional[float] = None
    spectral_flatness_std: Optional[float] = None

    spectral_rolloff_mean: Optional[float] = None
    spectral_rolloff_std: Optional[float] = None


# -- Quefrency Domain Features --
# MFCC_1-4
class MFCC_1_4(BaseModel):
    mfcc_1_mean: Optional[float] = None
    mfcc_1_std: Optional[float] = None
    mfcc_2_mean: Optional[float] = None
    mfcc_2_std: Optional[float] = None
    mfcc_3_mean: Optional[float] = None
    mfcc_3_std: Optional[float] = None
    mfcc_4_mean: Optional[float] = None
    mfcc_4_std: Optional[float] = None


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
