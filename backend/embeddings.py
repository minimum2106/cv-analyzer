import openai
import numpy as np

async def get_openai_embeddings(texts, model="text-embedding-ada-002"):
    """Get embeddings for the provided lines using OpenAI"""
    try:
        response = await openai.embeddings.create(
            input=texts, model=model
        )
        return [np.array(e["embedding"]) for e in response["data"]]
    except Exception as e:
        raise RuntimeError(f"Failed to get embeddings: {str(e)}")
