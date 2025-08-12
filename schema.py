from typing import List
from pydantic import BaseModel

class SwapFaceRequest(BaseModel):
    generation_id: str
    source_indices: List[int]
    target_indices: List[int]