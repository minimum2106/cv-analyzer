from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BeforeValidator, PlainSerializer
import numpy as np
from backend.services import OpenAIService, GroqService, ClaudeService
from typing import List, Annotated
import ast
import re


from backend.embeddings import get_openai_embeddings
from backend.prompts import (
    SIMILARITY_REASONING_PROMPT,
    JOB_REQUIREMENT_EXTRACTING_PROMPT,
)

app = FastAPI(title="EasyForm Backend API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


THRESHOLD = 0.7


def get_llm_service(model_type: str):
    if model_type == "openai":
        return OpenAIService()
    elif model_type == "groq":
        return GroqService()
    elif model_type == "claude":
        return ClaudeService()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


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


@app.post("/generate_job_lines", response_model=JobLinesResponse)
async def generate_job_lines(request: JobLinesRequest):
    llm_service = get_llm_service(request.model_type)
    if not llm_service:
        raise ValueError(f"Unsupported model type: {request.model_type}")

    job_lines = llm_service.call_api(
        system_prompt=JOB_REQUIREMENT_EXTRACTING_PROMPT,
        prompt="This is the job description: \n" + request.job_description,
    )

    return JobLinesResponse(job_lines=job_lines)


@app.post("/get_similarity_matrix", response_model=SimilarityMatrixResponse)
async def get_similarity_matrix(request: SimilarityMatrixRequest):
    cv_embeddings = []
    for line in request.cv_lines:
        cv_embedding = get_openai_embeddings(line)
        cv_embeddings.append(cv_embedding)

    job_embeddings = []
    for line in request.job_lines:
        job_embedding = get_openai_embeddings(line)
        job_embeddings.append(job_embedding)

    job_embeddings = np.array(job_embeddings).T  # Transpose for dot product calculation
    similarity_matrix = np.matmul(np.array(cv_embeddings), job_embeddings)
    similarity_matrix = similarity_matrix / (
        np.linalg.norm(cv_embeddings, axis=1)[:, None]
        * np.linalg.norm(job_embeddings, axis=0)
    )
    similarity_matrix = np.clip(
        similarity_matrix, 0, 1
    )  # Ensure values are between 0 and 1

    return SimilarityMatrixResponse(matrix=similarity_matrix)


@app.post("/explain_match", response_model=ExplainMatchResponse)
async def explain_match(request: ExplainMatchRequest):
    llm_service = get_llm_service(request.model_type)
    if not llm_service:
        raise ValueError(f"Unsupported model type: {request.model_type}")

    explanations = []
    for indice in request.filtered_indices:
        cv_index, job_index = indice
        if cv_index >= len(request.cv_lines) or job_index >= len(request.job_lines):
            raise ValueError("Index out of bounds for CV or Job lines")

        cv_line = request.cv_lines[cv_index]
        job_line = request.job_lines[job_index]

        explanation = llm_service.call_api(
            system_prompt=SIMILARITY_REASONING_PROMPT,
            prompt=f"""
                Explain the similarity between the following CV line and Job line:
                CV Line: {cv_line}
                Job Line: {job_line}
            """,
        )
        explanation = explanation.strip()
        match = re.search(r"<Conclusion>(.*?)</Conclusion>", explanation, re.DOTALL)
        if match:
            conclusion_text = match.group(1).strip()
        else:
            conclusion_text = "No conclusion found."

        explanations.append(conclusion_text)

    return JSONResponse(content={"explanations": explanations})
