from pydantic import BaseModel
from typing import Optional, Union, Literal


class VoiceModel(BaseModel):
    clip_id: Union[str, int]
    voice_name: str
    voice_age: Optional[int] = None
    voice_age_group: Literal[
        "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"
    ]
    voice_gender: Optional[Literal["m", "f"]]
