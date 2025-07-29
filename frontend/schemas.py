from pydantic import BaseModel
from typing import Tuple, List, Optional

class JobReqInfo(BaseModel):
    position: Tuple[float, float, float, float]
    text: str
    explanation: Optional[str] = ""
    

class CVLineInfo(BaseModel):
    position: Tuple[float, float, float, float]
    text: str
    connected_job_reqs: Optional[List[JobReqInfo]] = []

