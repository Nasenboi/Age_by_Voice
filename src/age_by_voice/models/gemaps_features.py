from pydantic import BaseModel, Field
from typing import Optional, Union
from opensmile import FeatureSet, FeatureLevel
import pandas as pd
from .features_model import FeaturesModel


class GeMAPS_Features(FeaturesModel):
    F0semitoneFrom27_5Hz_sma3nz_amean: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_amean"
    )
    F0semitoneFrom27_5Hz_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_stddevNorm"
    )
    F0semitoneFrom27_5Hz_sma3nz_percentile20_0: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_percentile20.0"
    )
    F0semitoneFrom27_5Hz_sma3nz_percentile50_0: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_percentile50.0"
    )
    F0semitoneFrom27_5Hz_sma3nz_percentile80_0: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_percentile80.0"
    )
    F0semitoneFrom27_5Hz_sma3nz_pctlrange0_2: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"
    )
    F0semitoneFrom27_5Hz_sma3nz_meanRisingSlope: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope"
    )
    F0semitoneFrom27_5Hz_sma3nz_stddevRisingSlope: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope"
    )
    F0semitoneFrom27_5Hz_sma3nz_meanFallingSlope: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope"
    )
    F0semitoneFrom27_5Hz_sma3nz_stddevFallingSlope: Optional[float] = Field(
        alias="F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope"
    )
    loudness_sma3_amean: Optional[float] = Field(alias="loudness_sma3_amean")
    loudness_sma3_stddevNorm: Optional[float] = Field(alias="loudness_sma3_stddevNorm")
    loudness_sma3_percentile20_0: Optional[float] = Field(
        alias="loudness_sma3_percentile20.0"
    )
    loudness_sma3_percentile50_0: Optional[float] = Field(
        alias="loudness_sma3_percentile50.0"
    )
    loudness_sma3_percentile80_0: Optional[float] = Field(
        alias="loudness_sma3_percentile80.0"
    )
    loudness_sma3_pctlrange0_2: Optional[float] = Field(
        alias="loudness_sma3_pctlrange0-2"
    )
    loudness_sma3_meanRisingSlope: Optional[float] = Field(
        alias="loudness_sma3_meanRisingSlope"
    )
    loudness_sma3_stddevRisingSlope: Optional[float] = Field(
        alias="loudness_sma3_stddevRisingSlope"
    )
    loudness_sma3_meanFallingSlope: Optional[float] = Field(
        alias="loudness_sma3_meanFallingSlope"
    )
    loudness_sma3_stddevFallingSlope: Optional[float] = Field(
        alias="loudness_sma3_stddevFallingSlope"
    )
    spectralFlux_sma3_amean: Optional[float] = Field(alias="spectralFlux_sma3_amean")
    spectralFlux_sma3_stddevNorm: Optional[float] = Field(
        alias="spectralFlux_sma3_stddevNorm"
    )
    mfcc1_sma3_amean: Optional[float] = Field(alias="mfcc1_sma3_amean")
    mfcc1_sma3_stddevNorm: Optional[float] = Field(alias="mfcc1_sma3_stddevNorm")
    mfcc2_sma3_amean: Optional[float] = Field(alias="mfcc2_sma3_amean")
    mfcc2_sma3_stddevNorm: Optional[float] = Field(alias="mfcc2_sma3_stddevNorm")
    mfcc3_sma3_amean: Optional[float] = Field(alias="mfcc3_sma3_amean")
    mfcc3_sma3_stddevNorm: Optional[float] = Field(alias="mfcc3_sma3_stddevNorm")
    mfcc4_sma3_amean: Optional[float] = Field(alias="mfcc4_sma3_amean")
    mfcc4_sma3_stddevNorm: Optional[float] = Field(alias="mfcc4_sma3_stddevNorm")
    jitterLocal_sma3nz_amean: Optional[float] = Field(alias="jitterLocal_sma3nz_amean")
    jitterLocal_sma3nz_stddevNorm: Optional[float] = Field(
        alias="jitterLocal_sma3nz_stddevNorm"
    )
    shimmerLocaldB_sma3nz_amean: Optional[float] = Field(
        alias="shimmerLocaldB_sma3nz_amean"
    )
    shimmerLocaldB_sma3nz_stddevNorm: Optional[float] = Field(
        alias="shimmerLocaldB_sma3nz_stddevNorm"
    )
    HNRdBACF_sma3nz_amean: Optional[float] = Field(alias="HNRdBACF_sma3nz_amean")
    HNRdBACF_sma3nz_stddevNorm: Optional[float] = Field(
        alias="HNRdBACF_sma3nz_stddevNorm"
    )
    logRelF0_H1_H2_sma3nz_amean: Optional[float] = Field(
        alias="logRelF0-H1-H2_sma3nz_amean"
    )
    logRelF0_H1_H2_sma3nz_stddevNorm: Optional[float] = Field(
        alias="logRelF0-H1-H2_sma3nz_stddevNorm"
    )
    logRelF0_H1_A3_sma3nz_amean: Optional[float] = Field(
        alias="logRelF0-H1-A3_sma3nz_amean"
    )
    logRelF0_H1_A3_sma3nz_stddevNorm: Optional[float] = Field(
        alias="logRelF0-H1-A3_sma3nz_stddevNorm"
    )
    F1frequency_sma3nz_amean: Optional[float] = Field(alias="F1frequency_sma3nz_amean")
    F1frequency_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F1frequency_sma3nz_stddevNorm"
    )
    F1bandwidth_sma3nz_amean: Optional[float] = Field(alias="F1bandwidth_sma3nz_amean")
    F1bandwidth_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F1bandwidth_sma3nz_stddevNorm"
    )
    F1amplitudeLogRelF0_sma3nz_amean: Optional[float] = Field(
        alias="F1amplitudeLogRelF0_sma3nz_amean"
    )
    F1amplitudeLogRelF0_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F1amplitudeLogRelF0_sma3nz_stddevNorm"
    )
    F2frequency_sma3nz_amean: Optional[float] = Field(alias="F2frequency_sma3nz_amean")
    F2frequency_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F2frequency_sma3nz_stddevNorm"
    )
    F2bandwidth_sma3nz_amean: Optional[float] = Field(alias="F2bandwidth_sma3nz_amean")
    F2bandwidth_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F2bandwidth_sma3nz_stddevNorm"
    )
    F2amplitudeLogRelF0_sma3nz_amean: Optional[float] = Field(
        alias="F2amplitudeLogRelF0_sma3nz_amean"
    )
    F2amplitudeLogRelF0_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F2amplitudeLogRelF0_sma3nz_stddevNorm"
    )
    F3frequency_sma3nz_amean: Optional[float] = Field(alias="F3frequency_sma3nz_amean")
    F3frequency_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F3frequency_sma3nz_stddevNorm"
    )
    F3bandwidth_sma3nz_amean: Optional[float] = Field(alias="F3bandwidth_sma3nz_amean")
    F3bandwidth_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F3bandwidth_sma3nz_stddevNorm"
    )
    F3amplitudeLogRelF0_sma3nz_amean: Optional[float] = Field(
        alias="F3amplitudeLogRelF0_sma3nz_amean"
    )
    F3amplitudeLogRelF0_sma3nz_stddevNorm: Optional[float] = Field(
        alias="F3amplitudeLogRelF0_sma3nz_stddevNorm"
    )
    alphaRatioV_sma3nz_amean: Optional[float] = Field(alias="alphaRatioV_sma3nz_amean")
    alphaRatioV_sma3nz_stddevNorm: Optional[float] = Field(
        alias="alphaRatioV_sma3nz_stddevNorm"
    )
    hammarbergIndexV_sma3nz_amean: Optional[float] = Field(
        alias="hammarbergIndexV_sma3nz_amean"
    )
    hammarbergIndexV_sma3nz_stddevNorm: Optional[float] = Field(
        alias="hammarbergIndexV_sma3nz_stddevNorm"
    )
    slopeV0_500_sma3nz_amean: Optional[float] = Field(alias="slopeV0-500_sma3nz_amean")
    slopeV0_500_sma3nz_stddevNorm: Optional[float] = Field(
        alias="slopeV0-500_sma3nz_stddevNorm"
    )
    slopeV500_1500_sma3nz_amean: Optional[float] = Field(
        alias="slopeV500-1500_sma3nz_amean"
    )
    slopeV500_1500_sma3nz_stddevNorm: Optional[float] = Field(
        alias="slopeV500-1500_sma3nz_stddevNorm"
    )
    spectralFluxV_sma3nz_amean: Optional[float] = Field(
        alias="spectralFluxV_sma3nz_amean"
    )
    spectralFluxV_sma3nz_stddevNorm: Optional[float] = Field(
        alias="spectralFluxV_sma3nz_stddevNorm"
    )
    mfcc1V_sma3nz_amean: Optional[float] = Field(alias="mfcc1V_sma3nz_amean")
    mfcc1V_sma3nz_stddevNorm: Optional[float] = Field(alias="mfcc1V_sma3nz_stddevNorm")
    mfcc2V_sma3nz_amean: Optional[float] = Field(alias="mfcc2V_sma3nz_amean")
    mfcc2V_sma3nz_stddevNorm: Optional[float] = Field(alias="mfcc2V_sma3nz_stddevNorm")
    mfcc3V_sma3nz_amean: Optional[float] = Field(alias="mfcc3V_sma3nz_amean")
    mfcc3V_sma3nz_stddevNorm: Optional[float] = Field(alias="mfcc3V_sma3nz_stddevNorm")
    mfcc4V_sma3nz_amean: Optional[float] = Field(alias="mfcc4V_sma3nz_amean")
    mfcc4V_sma3nz_stddevNorm: Optional[float] = Field(alias="mfcc4V_sma3nz_stddevNorm")
    alphaRatioUV_sma3nz_amean: Optional[float] = Field(
        alias="alphaRatioUV_sma3nz_amean"
    )
    hammarbergIndexUV_sma3nz_amean: Optional[float] = Field(
        alias="hammarbergIndexUV_sma3nz_amean"
    )
    slopeUV0_500_sma3nz_amean: Optional[float] = Field(
        alias="slopeUV0-500_sma3nz_amean"
    )
    slopeUV500_1500_sma3nz_amean: Optional[float] = Field(
        alias="slopeUV500-1500_sma3nz_amean"
    )
    spectralFluxUV_sma3nz_amean: Optional[float] = Field(
        alias="spectralFluxUV_sma3nz_amean"
    )
    loudnessPeaksPerSec: Optional[float]
    VoicedSegmentsPerSec: Optional[float]
    MeanVoicedSegmentLengthSec: Optional[float]
    StddevVoicedSegmentLengthSec: Optional[float]
    MeanUnvoicedSegmentLength: Optional[float]
    StddevUnvoicedSegmentLength: Optional[float]
    equivalentSoundLevel_dBp: Optional[float]


def parse_gemaps_features(df, clip_id: Union[str, int]) -> GeMAPS_Features:
    """
    Parses a dataframe to create a FeaturesModel instance.
    Args:
        df (pd.DataFrame): DataFrame containing the features data.
        clip_id (Union[str, int]): Identifier for the audio clip.

    Returns:
        FeaturesModel: An instance of FeaturesModel populated with the parsed data.
    """
    # Convert the DataFrame to a dictionary
    feature_data = df.iloc[0].to_dict()
    feature_data["clip_id"] = clip_id

    return GeMAPS_Features(**feature_data)
