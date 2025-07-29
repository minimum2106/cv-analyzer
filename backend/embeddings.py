import openai
import numpy as np


def get_openai_embeddings(texts, model="text-embedding-3-large"):
    """Get embeddings for the provided lines using OpenAI"""
    try:
        response = openai.embeddings.create(input=texts, model=model)
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Failed to get embeddings: {str(e)}")
