from typing import Dict
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from collections import defaultdict

# Load environment variables using os.environ
openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

if not openai_api_key:
    raise RuntimeError("OpenAI API key is missing. Please set it in the environment variables.")

if not pinecone_api_key:
    raise RuntimeError("Pinecone API key is missing. Please set it in the environment variables.")

try:
    import openai
    openai.api_key = openai_api_key
except ModuleNotFoundError:
    raise ImportError("The 'openai' module is not installed. Please install it using 'pip install openai'.")

try:
    from pinecone import Pinecone
    index_name = "rags"
    pc1 = Pinecone(api_key=pinecone_api_key)
    index = pc1.Index(index_name)
except ModuleNotFoundError:
    raise ImportError("The 'pinecone' module is not installed. Please install it using 'pip install pinecone-client'.")

# Initialize FastAPI app
app = FastAPI()

# Initialize OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Session store to maintain user-specific context
session_store: Dict[str, str] = defaultdict(str)

# Validate the token and extract user ID
def validate_token(token: str = Depends(oauth2_scheme)) -> str:
    # Dummy logic to extract user ID from token (replace with real validation)
    user_id = "user_" + token  # For testing purposes, the token itself acts as a user ID
    if not user_id:
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return user_id

# Query model for user input
class QueryModel(BaseModel):
    query: str  # User query to search the context

@app.post("/query")
async def get_response(
    query_data: QueryModel,
    user_id: str = Depends(validate_token),
):
    try:
        # Step 1: Retrieve user's last context
        last_context = session_store.get(user_id, "")

        # Ensure OpenAI and Pinecone modules are working
        if not hasattr(openai, "Embedding"):
            raise RuntimeError("OpenAI module is not functioning correctly. Check the installation.")

        if not hasattr(index, "query"):
            raise RuntimeError("Pinecone index is not functioning correctly. Check the setup.")

        # Step 2: Convert query to embeddings
        res = openai.Embedding.create(
            input=[query_data.query], model="text-embedding-ada-002"
        )
        embedding = res["data"][0]["embedding"]

        # Step 3: Search for matching vectors in Pinecone
        results = index.query(vector=embedding, top_k=3, include_metadata=True)
        matches = results.get("matches", [])
        sources = list(set(match["metadata"].get("source", "") for match in matches if "source" in match["metadata"]))

        # Step 4: Extract context from search results
        context = [match["metadata"].get("text", "") for match in results.get("matches", [])]

        # Combine last context with new search context
        combined_context = f"{last_context}\n{chr(10).join(context)}".strip()

        if not combined_context:
            raise HTTPException(status_code=404, detail="No matching context found.")

        # Step 5: Combine combined context with the user's query
        full_input = f"Context:\n{combined_context}\n\nUser Query:\n{query_data.query}"

        # Step 6: Get GPT-4 response
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_input}
            ],
            max_tokens=500,
            temperature=0.7,
        )

        # Step 7: Extract response text
        response_text = gpt_response["choices"][0]["message"]["content"].strip()

        # Step 8: Update the session store with the new context
        session_store[user_id] = f"{combined_context}\n{response_text}".strip()

        # Return the GPT-4 response
        return {"response": response_text, "source": sources}

    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
