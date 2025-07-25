from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pymupdf
from pymupdf import Document
from pydantic import BaseModel, BeforeValidator, PlainSerializer
import openai
import numpy as np
import os
from backend.services import OpenAIService, GroqService, ClaudeService
from typing import List, Annotated
import ast


from backend.embeddings import get_openai_embeddings
from backend.prompts import (
    similarity_reasoning_prompt,
    job_requirement_extracting_prompt,
)
from backend.extract_doc import (
    get_lines_with_coords,
    get_bullets_from_doc,
    merge_lines_by_bullets,
    LineWithCoords,
)

app = FastAPI(title="EasyForm Backend API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


openai.api_key = os.getenv("OPENAI_API_KEY")
THRESHOLD = 0.7
MODELS = {
    "openai": OpenAIService,
    "groq": GroqService,
    "claude": ClaudeService,
}

# Pydantic models for request/response


class CVTextRequest(BaseModel):
    model_type: str
    doc: Document

    class Config:
        arbitrary_types_allowed = True


class CVTextResponse(BaseModel):
    cv_lines: list[LineWithCoords]


class JobLinesRequest(BaseModel):
    model_type: str
    job_description: str


class JobLinesResponse(BaseModel):
    job_lines: List[LineWithCoords]


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
    explanation: str


@app.post("/extract_cv_text", response_model=CVTextResponse)
async def extract_cv_text(request: CVTextRequest):
    lines_with_coords, fontsize = get_lines_with_coords(request.doc)
    bullets = get_bullets_from_doc(request.doc, fontsize)
    merged_lines = merge_lines_by_bullets(lines_with_coords, bullets)

    return CVTextResponse(cv_lines=merged_lines)


@app.post("/extract_job_lines", response_model=JobLinesResponse)
async def extract_job_lines(request: JobLinesRequest):
    llm_service = MODELS.get(request.model_type)
    if not llm_service:
        raise ValueError(f"Unsupported model type: {request.model_type}")

    job_requirements = llm_service.call_api(
        system_prompt=job_requirement_extracting_prompt(request.job_description)
    )

    doc = pymupdf.open()
    doc.new_page()
    page = doc[0]
    page.insert_text((50, 50), job_requirements, fontsize=12)

    # like before, extract lines and coordinates
    job_lines_with_coords, job_fontsize = get_lines_with_coords(doc)
    job_bullets = get_bullets_from_doc(doc, job_fontsize)

    # Merge lines with coordinates
    job_merged_lines = merge_lines_by_bullets(job_lines_with_coords, job_bullets)

    return JobLinesResponse(job_lines=job_merged_lines)


@app.post("/get_similarity_matrix", response_model=SimilarityMatrixResponse)
async def get_similarity_matrix(request: SimilarityMatrixRequest):
    cv_embeddings = await get_openai_embeddings(request.cv_lines)
    job_embeddings = get_openai_embeddings(request.job_lines)

    job_embeddings = job_embeddings.T  # Transpose for dot product calculation
    similarity_matrix = np.dot(cv_embeddings, job_embeddings)
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
    llm_service = MODELS.get(request.model_type)
    if not llm_service:
        raise ValueError(f"Unsupported model type: {request.model_type}")

    explainations = []
    for indice in request.filtered_indices:
        cv_index, job_index = indice
        if cv_index >= len(request.cv_lines) or job_index >= len(request.job_lines):
            raise ValueError("Index out of bounds for CV or Job lines")

        cv_line = request.cv_lines[cv_index]
        job_line = request.job_lines[job_index]

        explaination = llm_service.call_api(
            system_prompt=similarity_reasoning_prompt(cv_line, job_line)
        )
        explaination = explaination.strip()
        explainations.append(explaination)

    return JSONResponse(content={"explanations": explainations})
