from typing import List, Annotated
from pydantic import BaseModel, BeforeValidator, PlainSerializer

import numpy as np
import ast

class JobLinesRequest(BaseModel):
    model_type: str
    job_description: str


class JobLinesResponse(BaseModel):
    job_lines: str


class SimilarityMatrixRequest(BaseModel):
    cv_lines: List[str]
    job_lines: List[str]


# https://www.flowphysics.com/2024/02/12/numpy-arrays-in-pydantic.html
def nd_array_before_validator(x):
    # custom before validation logic
    if isinstance(x, str):
        x_list = ast.literal_eval(x)
        x = np.array(x_list)
    if isinstance(x, List):
        x = np.array(x)
    return x


def nd_array_serializer(x):
    # custom serialization logic
    return x.tolist()
    # return np.array2string(x,separator=',', threshold=sys.maxsize)


MyNumPyArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_before_validator),
    PlainSerializer(nd_array_serializer, return_type=List),
]


class SimilarityMatrixResponse(BaseModel):
    matrix: MyNumPyArray

    class Config:
        arbitrary_types_allowed = True


class MatchItem(BaseModel):
    cv_line: str
    job_line: str
    similarity: float


class ExplainMatchRequest(BaseModel):
    model_type: str
    cv_lines: List[str]
    job_lines: List[str]
    filtered_indices: List[List[int]]  # List of [cv_index, job_index]


class ExplainMatchResponse(BaseModel):
    explanations: List[str]