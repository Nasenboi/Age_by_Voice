from pydantic import BaseModel
from typing import Optional, Union
from opensmile import FeatureSet, FeatureLevel
import pandas as pd

FEATURE_SET = FeatureSet.eGeMAPSv02
FEATURE_LEVEL = FeatureLevel.Functionals


# Replaced any "." with "_" in the feature names
# F0semitoneFrom27.5Hz<...> with F0semitoneFrom27_5Hz<...>
class FeaturesModel(BaseModel):
    clip_id: Optional[Union[str, int]] = None
