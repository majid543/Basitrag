from typing import Dict
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from collections import defaultdict

# Load environment variables using os.environ
openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
render_api_token = os.environ.get("RENDER_API_TOKEN")

if not openai_api_key:
    raise RuntimeError("OpenAI API key is missing. Please set it in the environment variables.")

if not pinecone_api_key:
    raise RuntimeError("Pinecone API key is missing. Please set it in the environment variables.")

if not render_api_token:
    raise RuntimeError("Render API token is missing. Please set it in the environment variables.")

try:
    import openai
    openai.api_key = openai_api_key
except ModuleNotFoundError:
    raise ImportError("The 'openai' module is not installed. Please install it using 'pip install openai'.")

try:
    import pinecone
    pinecone.init(api_key=pinecone_api_key)
    index_name = "rags"
    if index_name not in pinecone.list_indexes():
        raise RuntimeError(f"Pinecone index '{index_name}' not found.")
    index = pinecone.Index(index_name)
except ModuleNotFoundError:
    raise ImportError("The 'pinecone-client' module is not installed. Please install it using 'pip install pinecone-client'.")

# Initialize FastAPI app
app = FastAPI()

# Initialize OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
security = HTTPBearer()

# Session store to maintain user-specific context (using a list to store multiple contexts)
session_store: Dict[str, list] = defaultdict(list)

# Validate the Render API token for authorization and extract session/user ID
def validate_tokens(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
    session_token: str = Depends(oauth2_scheme)  # Session token for user identification
) -> str:
    # Validate Render API token for authentication
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != render_api_token:
            raise HTTPException(status_code=403, detail="Invalid Render API token")
    else:
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")
    
    # Validate session token (this could be a simple check for now)
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token is required")
    
    return session_token  # Return the session token for further use

# Query model for user input
class QueryModel(BaseModel):
    query: str  # User query to search the context

@app.post("/query")
async def get_response(query_data: QueryModel, session_token: str = Depends(validate_tokens)):
    try:
        # Step 1: Retrieve the last context using the session token
        last_context = session_store.get(session_token, [])
        
        # If there are no stored contexts, use an empty string as a fallback
        last_context = last_context[-1] if last_context else ""
        print(last_context)
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
        context = [match["metadata"].get("text", "") for match in matches]

        # Combine last context with new search context
        combined_context = (
            f"This is the last conversation of the user for reference:\n{last_context}\n\n"
            f"This is the context that was retrieved as per the new user query:\n{chr(10).join(context)}"
        ).strip()

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

        # Step 8: Update the session store with the new context (append to the list)
        session_store[session_token].append(response_text)

        # Return the GPT-4 response
        return {"response": response_text, "source": sources}

    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

