import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# uvicorn app:app --host 0.0.0.0 --port 10000
app = FastAPI()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
    prompt: str  # User input prompt to pass to GPT-4


@app.post("/")
async def get_response(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    # Step 1: Convert query to embeddings
    res = openai_client.embeddings.create(
        input=[query_data.query], model="text-embedding-ada-002"
    )
    embedding = res.data[0].embedding

    # Step 2: Search for matching Vectors in Pinecone
    results = index.query(vector=embedding, top_k=3, include_metadata=True).to_dict()

    # Step 3: Extract context from search results
    context = [match["metadata"]["text"] for match in results["matches"]]

    # Step 4: Combine context with the user's prompt
    full_input = f"Context:\n{'\n'.join(context)}\n\nUser Prompt:\n{query_data.prompt}"

    # Step 5: Get GPT-4 response
    gpt_response = openai_client.completions.create(
        model="gpt-4",
        prompt=full_input,
        max_tokens=500,
        temperature=0.7,
    )

    # Step 6: Return GPT-4 response
    return {"response": gpt_response.choices[0].text.strip()}