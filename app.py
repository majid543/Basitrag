import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
from pinecone import Pinecone

load_dotenv()

# uvicorn app:app --host 0.0.0.0 --port 10000
app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
index_name = "rags"
pc1 = Pinecone(api_key=pinecone_api_key)
index = pc1.Index(index_name)

# Middleware to secure HTTP endpoint
security = HTTPBearer()


def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
):
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != os.getenv("RENDER_API_TOKEN"):
            raise HTTPException(status_code=403, detail="Invalid token")
    else:
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")


class QueryModel(BaseModel):
    query: str  # User query to search the context


@app.post("/")
async def get_response(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        # Step 1: Convert query to embeddings
        res = openai.Embedding.create(
            input=[query_data.query], model="text-embedding-ada-002"
        )
        embedding = res["data"][0]["embedding"]

        # Step 2: Search for matching vectors in Pinecone
        results = index.query(vector=embedding, top_k=3, include_metadata=True)

        matches = results.get("matches", [])
        sources = list(set(match["metadata"].get("source", "") for match in matches if "source" in match["metadata"]))

        # Step 3: Extract context from search results
        context = [match["metadata"].get("text", "") for match in results.get("matches", [])]

        # Check if context is empty
        if not context:
            raise HTTPException(status_code=404, detail="No matching context found.")

        # Step 4: Combine context with the user's query
        full_input = f"Context:\n{chr(10).join(context)}\n\nUser Query:\n{query_data.query}"

        # Step 5: Get GPT-4 response
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_input}
            ],
            max_tokens=500,
            temperature=0.7,
        )

        # Step 6: Extract response text
        response_text = gpt_response["choices"][0]["message"]["content"].strip()

        # Return the GPT-4 response
        return {"response": response_text,"source": sources}

    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
