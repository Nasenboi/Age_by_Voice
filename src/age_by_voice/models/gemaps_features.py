from pydantic import BaseModel
from typing import Optional, Union
from opensmile import FeatureSet, FeatureLevel
import pandas as pd
from .features_model import FeaturesModel


class GeMAPS_Features(FeaturesModel):
    F0semitoneFrom27_5Hz_sma3nz_amean: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_stddevNorm: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_percentile20_0: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_percentile50_0: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_percentile80_0: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_pctlrange0_2: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_meanRisingSlope: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_stddevRisingSlope: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_meanFallingSlope: Optional[float]
    F0semitoneFrom27_5Hz_sma3nz_stddevFallingSlope: Optional[float]
    loudness_sma3_amean: Optional[float]
    loudness_sma3_stddevNorm: Optional[float]
    loudness_sma3_percentile20_0: Optional[float]
    loudness_sma3_percentile50_0: Optional[float]
    loudness_sma3_percentile80_0: Optional[float]
    loudness_sma3_pctlrange0_2: Optional[float]
    loudness_sma3_meanRisingSlope: Optional[float]
    loudness_sma3_stddevRisingSlope: Optional[float]
    loudness_sma3_meanFallingSlope: Optional[float]
    loudness_sma3_stddevFallingSlope: Optional[float]
    spectralFlux_sma3_amean: Optional[float]
    spectralFlux_sma3_stddevNorm: Optional[float]
    mfcc1_sma3_amean: Optional[float]
    mfcc1_sma3_stddevNorm: Optional[float]
    mfcc2_sma3_amean: Optional[float]
    mfcc2_sma3_stddevNorm: Optional[float]
    mfcc3_sma3_amean: Optional[float]
    mfcc3_sma3_stddevNorm: Optional[float]
    mfcc4_sma3_amean: Optional[float]
    mfcc4_sma3_stddevNorm: Optional[float]
    jitterLocal_sma3nz_amean: Optional[float]
    jitterLocal_sma3nz_stddevNorm: Optional[float]
    shimmerLocaldB_sma3nz_amean: Optional[float]
    shimmerLocaldB_sma3nz_stddevNorm: Optional[float]
    HNRdBACF_sma3nz_amean: Optional[float]
    HNRdBACF_sma3nz_stddevNorm: Optional[float]
    logRelF0_H1_H2_sma3nz_amean: Optional[float]
    logRelF0_H1_H2_sma3nz_stddevNorm: Optional[float]
    logRelF0_H1_A3_sma3nz_amean: Optional[float]
    logRelF0_H1_A3_sma3nz_stddevNorm: Optional[float]
    F1frequency_sma3nz_amean: Optional[float]
    F1frequency_sma3nz_stddevNorm: Optional[float]
    F1bandwidth_sma3nz_amean: Optional[float]
    F1bandwidth_sma3nz_stddevNorm: Optional[float]
    F1amplitudeLogRelF0_sma3nz_amean: Optional[float]
    F1amplitudeLogRelF0_sma3nz_stddevNorm: Optional[float]
    F2frequency_sma3nz_amean: Optional[float]
    F2frequency_sma3nz_stddevNorm: Optional[float]
    F2bandwidth_sma3nz_amean: Optional[float]
    F2bandwidth_sma3nz_stddevNorm: Optional[float]
    F2amplitudeLogRelF0_sma3nz_amean: Optional[float]
    F2amplitudeLogRelF0_sma3nz_stddevNorm: Optional[float]
    F3frequency_sma3nz_amean: Optional[float]
    F3frequency_sma3nz_stddevNorm: Optional[float]
    F3bandwidth_sma3nz_amean: Optional[float]
    F3bandwidth_sma3nz_stddevNorm: Optional[float]
    F3amplitudeLogRelF0_sma3nz_amean: Optional[float]
    F3amplitudeLogRelF0_sma3nz_stddevNorm: Optional[float]
    alphaRatioV_sma3nz_amean: Optional[float]
    alphaRatioV_sma3nz_stddevNorm: Optional[float]
    hammarbergIndexV_sma3nz_amean: Optional[float]
    hammarbergIndexV_sma3nz_stddevNorm: Optional[float]
    slopeV0_500_sma3nz_amean: Optional[float]
    slopeV0_500_sma3nz_stddevNorm: Optional[float]
    slopeV500_1500_sma3nz_amean: Optional[float]
    slopeV500_1500_sma3nz_stddevNorm: Optional[float]
    spectralFluxV_sma3nz_amean: Optional[float]
    spectralFluxV_sma3nz_stddevNorm: Optional[float]
    mfcc1V_sma3nz_amean: Optional[float]
    mfcc1V_sma3nz_stddevNorm: Optional[float]
    mfcc2V_sma3nz_amean: Optional[float]
    mfcc2V_sma3nz_stddevNorm: Optional[float]
    mfcc3V_sma3nz_amean: Optional[float]
    mfcc3V_sma3nz_stddevNorm: Optional[float]
    mfcc4V_sma3nz_amean: Optional[float]
    mfcc4V_sma3nz_stddevNorm: Optional[float]
    alphaRatioUV_sma3nz_amean: Optional[float]
    hammarbergIndexUV_sma3nz_amean: Optional[float]
    slopeUV0_500_sma3nz_amean: Optional[float]
    slopeUV500_1500_sma3nz_amean: Optional[float]
    spectralFluxUV_sma3nz_amean: Optional[float]
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
        df (pd.DataFrame): DataFrame containing the smile feature set with original "." in the names.
        clip_id (Union[str, int]): The clip ID to associate with the FeaturesModel.

    Returns:
        FeaturesModel: An instance of FeaturesModel populated with the parsed data.
    """
    # Map the dataframe columns to the FeaturesModel attributes
    feature_data = {
        col.replace(".", "_"): df[col].iloc[0] if col in df.columns else None
        for col in GeMAPS_Features.model_fields.keys()
    }
    feature_data["clip_id"] = clip_id

    return GeMAPS_Features(**feature_data)
