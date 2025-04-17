from pydantic import BaseModel
from typing import Optional, Union, Literal


class VoiceModel(BaseModel):
    clip_id: Optional[Union[str, int]] = None
    audio_file_name: Optional[str] = None
    voice_name: Optional[str] = None
    voice_age: Optional[int] = None
    voice_age_group: Optional[
        Literal["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
    ] = None
    voice_gender: Optional[Literal["m", "f"]] = None
